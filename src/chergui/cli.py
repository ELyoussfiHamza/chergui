"""Command line: `chergui run scenarios/merzouga_chergui.yaml`."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from rich.console import Console
from rich.logging import RichHandler

from . import scenario as scenario_mod
from .sim import SimError

console = Console()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chergui", description="Stress-test PX4 drones in simulated Sahara conditions.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="fly a scenario and produce a report")
    p_run.add_argument("scenario", help="path to a scenario YAML file")
    p_run.add_argument("--gui", action="store_true", help="also open the Gazebo window (for recording video)")
    p_run.add_argument("--no-report", action="store_true", help="only fly and collect logs")

    p_rep = sub.add_parser("report", help="(re)build the report for an existing run directory")
    p_rep.add_argument("run_dir")
    p_rep.add_argument("--no-ai", action="store_true", help="skip the AI-written summary")

    p_mis = sub.add_parser("mission", help="describe a mission in words; the AI plans it, tests it, and revises it")
    p_mis.add_argument("intent", nargs="?", help='e.g. "survey the dunes east of camp in a midday chergui, lose GPS on the way back"')
    p_mis.add_argument("--test", help="use an existing test.yaml instead of drafting one from the intent")
    p_mis.add_argument("--budget", type=int, default=4, help="maximum number of flights (default 4)")
    p_mis.add_argument("--yes", action="store_true", help="don't ask for approval of the drafted test")
    p_mis.add_argument("--gui", action="store_true", help="also open the Gazebo window")

    p_mrep = sub.add_parser("mission-report", help="(re)build the report for an existing mission directory")
    p_mrep.add_argument("mission_dir")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="%H:%M:%S",
                        handlers=[RichHandler(console=console, show_path=False, markup=False)])

    try:
        if args.cmd == "run":
            sc = scenario_mod.load(args.scenario)
            console.rule(f"[bold #f08a4b]CHERGUI[/] · {sc.name}")
            from .runner import run
            run_dir = asyncio.run(run(sc, gui=args.gui))
            console.print(f"\n[bold]run saved:[/] {run_dir}")
            if not args.no_report:
                _report(run_dir, ai=True)
        elif args.cmd == "report":
            from pathlib import Path
            _report(Path(args.run_dir), ai=not args.no_ai)
        elif args.cmd == "mission":
            return _mission(args)
        elif args.cmd == "mission-report":
            from pathlib import Path

            from .mission_report import build as build_mission_report
            console.print(f"[bold]mission report:[/] {build_mission_report(Path(args.mission_dir))}")
    except (scenario_mod.ScenarioError, SimError) as exc:
        console.print(f"[bold red]error:[/] {exc}")
        return 1
    except KeyboardInterrupt:
        console.print("[yellow]interrupted[/]")
        return 130
    return 0


def _show_test(test: dict) -> None:
    from rich.table import Table

    from .agent import PLACES
    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column(style="bold #f08a4b")
    t.add_column()
    env, task = test.get("environment", {}), test["task"]
    wind = env.get("wind", {})
    t.add_row("test", f"{test['name']}: {test.get('description', '')}")
    t.add_row("place", PLACES[test["place"]]["name"])
    t.add_row("weather", f"{env.get('air_temp_c', 25):.0f} °C, wind {wind.get('speed_ms', 0):.0f} m/s "
                         f"from {wind.get('from_deg', 90):.0f}°")
    if task["type"] == "survey":
        (cn, ce), (ln, wd) = task["area_center_ne_m"], task["area_size_m"]
        t.add_row("task", f"survey a {ln:.0f} × {wd:.0f} m area centred {cn:.0f} m north, {ce:.0f} m east of launch")
    else:
        n, e = task["target_ne_m"]
        t.add_row("task", f"deliver to {n:.0f} m north, {e:.0f} m east of launch and come back")
    for f in test.get("faults", []):
        when = (f"at {f['at_progress']:.0%} of the route" if "at_progress" in f
                else f"at t={f['at_s']:.0f} s" if "at_s" in f else f"after waypoint {f.get('at_waypoint')}")
        what = (f"gust {f['speed_ms']:.0f} m/s from {f.get('from_deg', 90):.0f}° for {f['duration_s']:.0f} s"
                if f["type"] == "gust" else f"GNSS blackout for {f['duration_s']:.0f} s")
        t.add_row("fault", f"{what}, {when}")
    for k, v in test["criteria"].items():
        t.add_row("rule" if v is not None else "measure", f"{k.replace('_', ' ')}: {'—' if v is None else v}")
    console.print(t)


def _mission(args) -> int:
    from datetime import datetime
    from pathlib import Path

    import yaml

    from . import agent, config
    from .mission_report import build as build_mission_report

    if not 1 <= args.budget <= 8:
        console.print("[bold red]error:[/] --budget must be between 1 and 8")
        return 2
    if not (args.intent or args.test):
        console.print("[bold red]error:[/] describe the mission, or pass --test test.yaml")
        return 2
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    scratch = config.RUNS_DIR / "missions"
    scratch.mkdir(parents=True, exist_ok=True)

    if args.test:
        test = yaml.safe_load(Path(args.test).read_text())
        errors = agent.check_test(test, scratch)
        if errors:
            console.print("[bold red]test rejected:[/] " + "; ".join(errors))
            return 1
    else:
        console.rule("[bold #f08a4b]CHERGUI[/] · drafting the test")
        with console.status("the AI is turning your description into a test…"):
            test = agent.draft_test(args.intent, scratch)
    _show_test(test)
    if not args.yes and console.input("\n[bold]Lock this test and start planning?[/] [y/N] ").strip().lower() != "y":
        console.print("[yellow]cancelled: nothing flew[/]")
        return 1

    mission_dir = scratch / f"{stamp}_{agent.slugify(test['name'])}"
    mission_dir.mkdir(parents=True)

    def on_event(kind, **kw):
        if kind == "plan":
            p = kw["plan"]
            console.rule(f"[bold #f08a4b]attempt {kw['n']}/{args.budget}[/]")
            console.print(f"[bold]plan:[/] {p.altitude_m:.0f} m, {p.speed_ms:.0f} m/s"
                          + (f", passes {p.line_spacing_m:.0f} m apart {p.line_direction}" if p.line_spacing_m else "")
                          + (f", PX4 {p.px4_params}" if p.px4_params else ""))
            console.print(f"[dim]{p.rationale}[/]")
        elif kind == "result":
            for v in kw["analysis"]["verdicts"]:
                mark = "[green]PASS[/]" if v["passed"] else "[red]FAIL[/]"
                val = "—" if v["value"] is None else f"{v['value']:.1f}"
                console.print(f"  {mark} {v['check']}: {val} {v['unit']} (limit {v['limit']})")
        elif kind == "gave_up":
            console.print(f"[yellow]the planner stopped:[/] {kw['reason']}")

    import asyncio
    data = asyncio.run(agent.run_mission(test, mission_dir, args.budget, gui=args.gui, on_event=on_event))
    o = data["outcome"]
    console.rule()
    msg = {"passed": f"[bold green]PASSED[/] on attempt {o.get('attempt')}",
           "gave_up": "[bold yellow]STOPPED[/]: the planner found nothing more it is allowed to change",
           "budget_spent": f"[bold red]NOT PASSED[/] within {args.budget} flights"}[o["status"]]
    console.print(msg)
    console.print(f"[bold]mission report:[/] {build_mission_report(mission_dir)}")
    return 0 if o["status"] == "passed" else 3


def _report(run_dir, ai: bool) -> None:
    try:
        from .report import build
    except ImportError:
        console.print("[dim]report module not built yet[/]")
        return
    path = build(run_dir, ai=ai)
    console.print(f"[bold]report:[/] {path}")


if __name__ == "__main__":
    sys.exit(main())
