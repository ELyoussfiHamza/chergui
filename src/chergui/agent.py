"""Intent in, verified mission out.

    human sentence ──► AI drafts the TEST ──► human approves (locked from here on)
                                                │
                ┌──────────────► AI writes a PLAN ──► fixed checks ──► fly in Chergui ──► fixed rules
                │                                                                           │
                └──────────── AI reads the verdicts and the evidence, revises ◄─── failed ──┘

Code drives the loop; the AI only proposes. It can't change the test, can't judge a flight, and the loop
always ends: all rules pass, the run budget is spent, or the AI says nothing it may change would help.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path

import yaml

from . import config, planning
from .planning import Plan
from .scenario import ScenarioError, load, parse_task
from .summary import _rounded

log = logging.getLogger("chergui.agent")

# Places are picked by name, never invented: a model will happily give a confident lat/lon in someone's
# backyard. Each entry was checked against satellite imagery.
PLACES = {
    "erg_chebbi": {"name": "Erg Chebbi, Merzouga", "lat": 31.1330, "lon": -3.9790, "alt_m": 760},
}

TEST_LIMITS = {  # keep tests physical and short enough to fly in real time
    "max_extent_m": 400, "max_wind_ms": 20, "max_gust_ms": 25, "max_gps_loss_s": 90, "max_flight_time_s": 900,
}

TEST_WRITER = f"""You turn a person's description of a drone flight test into a test definition for a PX4
simulation in the Moroccan Sahara. You write the TEST only (conditions, task, faults, pass/fail rules).
You do NOT plan how to fly: altitude, speed and settings are decided later by someone else.

Reply with a JSON object:
{{
  "name": "short title",
  "description": "one sentence: the mission purpose and what is being stressed",
  "place": one of {list(PLACES)},
  "environment": {{"air_temp_c": number, "wind": {{"speed_ms": number, "from_deg": number}}}},
  "task": either
     {{"type": "survey", "area_center_ne_m": [north, east], "area_size_m": [north-south length, east-west width]}}
     or {{"type": "delivery", "target_ne_m": [north, east]}},
  "faults": [ {{"type": "gust", "at_progress": 0-1, "duration_s": n, "speed_ms": n, "from_deg": n}},
              {{"type": "gps_loss", "at_progress": 0-1, "duration_s": n}} ],
  "criteria": {{ ... }}
}}

Rules:
- Positions are metres north/east of the launch point. Keep everything within {TEST_LIMITS['max_extent_m']} m of
  launch (the simulation runs in real time).
- Wind from_deg is where the wind comes FROM. The Chergui is a hot, dry easterly (from ~90 deg), often 6-12 m/s.
  Desert midday air is ~40-48 C. Max wind {TEST_LIMITS['max_wind_ms']} m/s, max gust {TEST_LIMITS['max_gust_ms']} m/s,
  GNSS loss at most {TEST_LIMITS['max_gps_loss_s']} s.
- Faults trigger with at_progress = fraction of the route completed (e.g. "on the way back" ~ 0.7).
- Only these fault types exist: gust, gps_loss. If the person asks for something else (motor failure, dust on
  the camera...), leave it out and say so in the description.
- criteria keys (use null to measure without judging):
  max_flight_time_s (required, <= {TEST_LIMITS['max_flight_time_s']}), allow_uncommanded_takeoff (false),
  max_touchdown_speed_ms (hard landing, typically 1.5), max_path_deviation_m (containment, typically 50),
  max_landing_distance_m (only if the drone must come back to launch, typically 10),
  max_estimate_error_m (usually null: during a GNSS loss the estimate always drifts),
  min_coverage_pct (survey, typically 90), max_target_miss_m (delivery, typically 5).
- Judge outcomes that matter for safety and for the mission's purpose."""

PLANNER = f"""You are the mission planner for a PX4 quadcopter (x500, downward camera, 70 deg field of view)
flying a locked test in a simulated Sahara. You choose HOW to fly; you cannot change the test.

You may set:
- altitude_m ({planning.LIMITS['altitude_m'][0]:.0f}-{planning.LIMITS['altitude_m'][1]:.0f}),
  speed_ms ({planning.LIMITS['speed_ms'][0]:.0f}-{planning.LIMITS['speed_ms'][1]:.0f})
- survey only: line_spacing_m (must not exceed the camera footprint, about 1.4 x altitude),
  line_direction ("north-south" or "east-west")
- px4_params, only from this allowlist (anything not set keeps the PX4 default):
""" + "\n".join(f"    {k}: {v['doc']} [{v['min']}-{v['max']}]" for k, v in planning.PX4_ALLOWLIST.items()) + """
Waypoints are generated from your choices; the drone returns to launch and lands after the task.

You get the test and the history of previous attempts: your plan, the deterministic verdicts, measurements
from the flight log (ground truth vs the drone's estimate), fault windows and PX4's event timeline.

Think like a flight-test engineer:
- Explain failures from the evidence (numbers, events, timings), not guesses. Say what you expect to change.
- Change few things at a time so the next flight teaches something; keep what worked.
- Without GNSS the drone cannot navigate to a position. It can only hold attitude, descend or land.
- Safety comes first: never try to keep the drone flying blind or postpone a failsafe to pass a rule.
  The failsafe settings can only be made stricter, and the checks will reject anything else.
- If you believe no allowed change can fix the remaining failures, set give_up to true and explain why.
  "This mission should not be flown in these conditions without other position aiding" is a valid answer.
- Be concise: diagnosis and changes at most 70 words each.

Reply with a JSON object:
{
  "diagnosis": "what the evidence says went wrong (or, for the first plan, the main risks you see)",
  "changes": "what you set or changed and the effect you expect",
  "plan": {"altitude_m": n, "speed_ms": n, "line_spacing_m": n|null, "line_direction": "...", "px4_params": {}},
  "give_up": false,
  "give_up_reason": ""
}"""

DEBRIEF = """You write the short debrief of an autonomous mission-planning session: an AI planned a drone mission,
a simulator tested it under a locked test, fixed rules judged each flight, and the AI revised its plan.
Use ONLY numbers in the JSON, rounded sensibly; never invent. Plain words, no field names, no hype, no emojis.
Reply with a JSON object:
{"headline": "max 18 words", "story": "3-5 sentences: what failed, what the AI changed and why, how it ended",
 "lessons": ["2-3 takeaways a drone engineer would care about"]}"""


def _ask(instructions: str, payload: dict) -> dict:
    from openai import OpenAI

    resp = OpenAI().responses.create(
        model=config.OPENAI_MODEL,
        instructions=instructions,
        input="Input as JSON. Reply with the JSON object described.\n" + json.dumps(_rounded(payload), default=float),
        text={"format": {"type": "json_object"}},
    )
    return json.loads(resp.output_text)


# --- the test -----------------------------------------------------------------------------------

def _default_plan(task) -> Plan:
    return Plan(altitude_m=30, speed_ms=5, line_spacing_m=30 if task.type == "survey" else None)


def compose(test: dict, plan: Plan) -> dict:
    """Test + plan → a full scenario dict the runner understands."""
    task = parse_task(test["task"])
    sc = {k: v for k, v in test.items() if k not in ("place", "intent")}
    sc["location"] = PLACES[test["place"]]
    sc["mission"] = {"altitude_m": plan.altitude_m, "speed_ms": plan.speed_ms, "return_home": True,
                     "waypoints_ne_m": [list(map(lambda x: round(x, 2), w)) for w in planning.waypoints(task, plan)]}
    sc["px4_params"] = dict(plan.px4_params)
    return sc


def check_test(test: dict, scratch: Path) -> list[str]:
    errors = []
    if test.get("place") not in PLACES:
        return [f"place must be one of {list(PLACES)}"]
    try:
        task = parse_task(test.get("task") or {})
    except ScenarioError as exc:
        return [str(exc)]
    pts = [task.target_ne_m] if task.type == "delivery" else [
        (task.area_center_ne_m[0] + sn * task.area_size_m[0] / 2, task.area_center_ne_m[1] + se * task.area_size_m[1] / 2)
        for sn in (-1, 1) for se in (-1, 1)]
    if max(abs(c) for p in pts for c in p) > TEST_LIMITS["max_extent_m"]:
        errors.append(f"the task reaches beyond {TEST_LIMITS['max_extent_m']} m from launch")
    wind = test.get("environment", {}).get("wind", {})
    if wind.get("speed_ms", 0) > TEST_LIMITS["max_wind_ms"]:
        errors.append(f"wind above {TEST_LIMITS['max_wind_ms']} m/s")
    for f in test.get("faults", []):
        if f.get("type") == "gust" and f.get("speed_ms", 0) > TEST_LIMITS["max_gust_ms"]:
            errors.append(f"gust above {TEST_LIMITS['max_gust_ms']} m/s")
        if f.get("type") == "gps_loss" and f.get("duration_s", 0) > TEST_LIMITS["max_gps_loss_s"]:
            errors.append(f"GNSS loss longer than {TEST_LIMITS['max_gps_loss_s']} s")
    if not 60 <= (test.get("criteria") or {}).get("max_flight_time_s", 0) <= TEST_LIMITS["max_flight_time_s"]:
        errors.append(f"criteria.max_flight_time_s must be between 60 and {TEST_LIMITS['max_flight_time_s']}")
    if errors:
        return errors
    probe = scratch / "_probe.yaml"  # full validation by the real loader, with a placeholder plan
    probe.write_text(yaml.safe_dump(compose(test, _default_plan(task)), sort_keys=False))
    try:
        load(probe)
    except (ScenarioError, TypeError) as exc:
        errors.append(str(exc))
    finally:
        probe.unlink(missing_ok=True)
    return errors


def draft_test(intent: str, scratch: Path) -> dict:
    payload = {"request": intent}
    for _ in range(3):
        test = _ask(TEST_WRITER, payload)
        errors = check_test(test, scratch)
        if not errors:
            test["intent"] = intent
            return test
        log.info("test draft rejected by checks: %s", "; ".join(errors))
        payload = {"request": intent, "previous_draft": test, "errors_to_fix": errors}
    raise ScenarioError("could not draft a valid test: " + "; ".join(errors))


# --- the plan / fly / revise loop ---------------------------------------------------------------

def _evidence(n: int, plan: Plan, analysis: dict) -> dict:
    m = analysis["metrics"]
    keep = ("max_estimate_error_m", "max_deviation_m", "landing_distance_m", "takeoffs", "px4_detected_takeoffs",
            "flight_time_s", "hardest_touchdown_ms", "height_estimate_lost_s", "coverage_pct", "target_miss_m")
    return {
        "attempt": n,
        "plan": plan.to_dict(),
        "passed": analysis["passed"],
        "verdicts": [{k: v[k] for k in ("check", "value", "limit", "unit", "passed")} for v in analysis["verdicts"]],
        "measurements": {k: m[k] for k in keep} | {
            "touchdowns": [{k: d[k] for k in ("t", "dist_from_home_m", "speed_ms")} for d in m["touchdowns"]]},
        "faults": analysis["faults"],
        "timeline": [e for e in analysis["events"] if e["kind"] in ("px4", "mode", "fault", "fault_end", "state")][:80],
        "run_outcome": analysis["run"]["outcome"],
    }


def propose_plan(test: dict, history: list[dict]) -> tuple[Plan | None, dict]:
    task = parse_task(test["task"])
    payload = {"test": {k: v for k, v in test.items() if k != "place"}, "attempts_so_far": history}
    for _ in range(3):
        out = _ask(PLANNER, payload)
        if out.get("give_up"):
            return None, out
        p = out.get("plan") or {}
        try:
            plan = Plan(altitude_m=float(p["altitude_m"]), speed_ms=float(p["speed_ms"]),
                        line_spacing_m=p.get("line_spacing_m"), line_direction=p.get("line_direction") or "north-south",
                        px4_params={k: v for k, v in (p.get("px4_params") or {}).items()},
                        rationale=f"{out.get('diagnosis', '')}\n\n{out.get('changes', '')}".strip())
            errors = planning.check(task, test["criteria"]["max_flight_time_s"], plan)
        except (KeyError, TypeError, ValueError) as exc:
            errors = [f"plan is malformed: {exc}"]
        if not errors:
            return plan, out
        log.info("plan rejected by pre-flight checks: %s", "; ".join(errors))
        payload = {**payload, "rejected_plan": p, "errors_to_fix": errors}
    raise ScenarioError("the planner could not produce a plan that passes the pre-flight checks")


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "mission"


async def run_mission(test: dict, mission_dir: Path, budget: int, gui: bool = False, on_event=None) -> dict:
    from .report import build as build_report
    from .runner import run

    say = on_event or (lambda kind, **kw: None)
    (mission_dir / "test.yaml").write_text(yaml.safe_dump(test, sort_keys=False, allow_unicode=True))
    history, attempts, outcome = [], [], None
    for n in range(1, budget + 1):
        plan, raw = propose_plan(test, history)
        if plan is None:
            outcome = {"status": "gave_up", "reason": raw.get("give_up_reason", "")}
            say("gave_up", reason=outcome["reason"])
            break
        say("plan", n=n, plan=plan)
        sc_path = mission_dir / f"attempt-{n}.yaml"
        sc_path.write_text(yaml.safe_dump(compose(test, plan), sort_keys=False, allow_unicode=True))
        run_dir = await run(load(sc_path), gui=gui, run_dir=mission_dir / f"attempt-{n}")
        build_report(run_dir, ai=False)
        analysis = json.loads((run_dir / "analysis.json").read_text())
        history.append(_evidence(n, plan, analysis))
        attempts.append({"n": n, "plan": plan.to_dict(), "dir": run_dir.name, "passed": analysis["passed"],
                         "diagnosis": raw.get("diagnosis", ""), "changes": raw.get("changes", ""),
                         "verdicts": analysis["verdicts"], "metrics": analysis["metrics"],
                         "faults": analysis["faults"], "plan_latlon": analysis["plan_latlon"],
                         "truth_latlon": analysis["series"]["truth_latlon"][::2],
                         "est_latlon": analysis["series"]["est_latlon"][::2]})
        say("result", n=n, analysis=analysis)
        _save(mission_dir, test, attempts, outcome)
        if analysis["passed"]:
            outcome = {"status": "passed", "attempt": n}
            break
    if outcome is None:
        outcome = {"status": "budget_spent", "budget": budget}

    debrief = None
    if attempts:
        try:
            debrief = _ask(DEBRIEF, {"test": test, "outcome": outcome, "attempts": history})
        except Exception as exc:  # the debrief is a nice-to-have
            log.warning("debrief failed: %s", exc)
    return _save(mission_dir, test, attempts, outcome, debrief)


def _save(mission_dir: Path, test, attempts, outcome, debrief=None) -> dict:
    data = {"test": test, "attempts": attempts, "outcome": outcome, "debrief": debrief,
            "place": PLACES[test["place"]], "created": datetime.now().isoformat(timespec="seconds")}
    (mission_dir / "mission.json").write_text(json.dumps(data, indent=1, default=float))
    return data
