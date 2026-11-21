"""Judge a flight from its log. Deterministic rules only: the same log always gets the same verdict.

The key trick: in simulation PX4 logs two positions side by side.
  vehicle_global_position              what the drone BELIEVED (its EKF estimate)
  vehicle_global_position_groundtruth  where it ACTUALLY was (straight from Gazebo)
Real flights never have the second one. That gap is exactly what a GPS blackout opens up.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from pyulog import ULog

from . import scenario as scenario_mod
from .planning import footprint_m

EARTH_R = 6378137.0
RATE_HZ = 5.0  # resample everything onto one 5 Hz timeline


@dataclass
class Verdict:
    check: str
    value: float | None
    limit: float | None
    unit: str
    passed: bool
    detail: str


def _to_en(lat, lon, lat0, lon0):
    """lat/lon (deg) → metres east/north of (lat0, lon0)."""
    n = np.radians(np.asarray(lat) - lat0) * EARTH_R
    e = np.radians(np.asarray(lon) - lon0) * EARTH_R * math.cos(math.radians(lat0))
    return e, n


def _dist_to_polyline(px, py, pts) -> np.ndarray:
    """Distance from each point (px, py) to the nearest segment of the route."""
    best = np.full(len(px), np.inf)
    for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy or 1e-9
        t = np.clip(((px - ax) * dx + (py - ay) * dy) / L2, 0, 1)
        best = np.minimum(best, np.hypot(px - (ax + t * dx), py - (ay + t * dy)))
    return best


def _windows(events, kind_start="fault", kind_end="fault_end"):
    """Pair each fault event with its end event → [(label, type, t_start_us, t_end_us)]."""
    out, open_ = [], []
    for e in events:
        if e["kind"] == kind_start:
            ftype = "gps_loss" if "GNSS" in e["text"] else "gust"
            open_.append((e["text"], ftype, e["px4_us"]))
        elif e["kind"] == kind_end and open_:
            label, ftype, t0 = open_.pop(0)
            out.append((label, ftype, t0, e["px4_us"]))
    out += [(label, ftype, t0, None) for label, ftype, t0 in open_]
    return out


def _coverage_pct(task, ge, gn, agl, airborne, cell_m=4.0) -> float:
    """Share of the survey rectangle that the camera footprint actually passed over (from ground truth)."""
    (cn, ce), (length, width) = task.area_center_ne_m, task.area_size_m
    nn = np.arange(cn - length / 2 + cell_m / 2, cn + length / 2, cell_m)
    ee = np.arange(ce - width / 2 + cell_m / 2, ce + width / 2, cell_m)
    cells_e, cells_n = (a.ravel() for a in np.meshgrid(ee, nn))
    m = airborne & (agl > 3.0)
    if not m.any():
        return 0.0
    half = footprint_m(agl[m], task.camera_hfov_deg) / 2  # square-ish footprint, half-width per sample
    seen = np.zeros(len(cells_e), bool)
    for x, y, h in zip(ge[m], gn[m], half):
        seen |= (np.abs(cells_e - x) <= h) & (np.abs(cells_n - y) <= h)
    return float(100 * seen.mean())


def analyze(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run.json").read_text())
    sc = scenario_mod.load(run_dir / "scenario.yaml")
    events = meta["events"]

    ulog = ULog(str(run_dir / "flight.ulg"), [
        "vehicle_global_position", "vehicle_global_position_groundtruth", "vehicle_land_detected",
        "vehicle_local_position",
    ])
    est = ulog.get_dataset("vehicle_global_position").data
    gt = ulog.get_dataset("vehicle_global_position_groundtruth").data
    land = ulog.get_dataset("vehicle_land_detected").data
    lpos = ulog.get_dataset("vehicle_local_position").data

    # --- time base: seconds since the mission started --------------------------------------
    t_mission = next((e["px4_us"] for e in events if e["text"] == "mission started"), None)
    landed = land["landed"].astype(bool)
    takeoffs_us = land["timestamp"][1:][landed[:-1] & ~landed[1:]]
    landings_us = land["timestamp"][1:][~landed[:-1] & landed[1:]]
    if t_mission is None:
        t_mission = int(takeoffs_us[0]) if len(takeoffs_us) else int(gt["timestamp"][0])
    t_end = int(landings_us[-1]) + 3_000_000 if len(landings_us) else int(gt["timestamp"][-1])
    t_start = t_mission - 3_000_000
    tl = np.arange(t_start, t_end, 1e6 / RATE_HZ)
    sec = lambda us: None if us is None else (us - t_mission) / 1e6  # noqa: E731

    # --- positions on the common timeline, as metres around home ---------------------------
    lat0, lon0 = meta["plan_latlon"][0] if meta["plan_latlon"] else (gt["lat"][0], gt["lon"][0])
    interp = lambda d, k: np.interp(tl, d["timestamp"], d[k])  # noqa: E731
    gt_lat, gt_lon, gt_alt = interp(gt, "lat"), interp(gt, "lon"), interp(gt, "alt")
    es_lat, es_lon, es_alt = interp(est, "lat"), interp(est, "lon"), interp(est, "alt")
    ge, gn = _to_en(gt_lat, gt_lon, lat0, lon0)
    ee, en = _to_en(es_lat, es_lon, lat0, lon0)
    home_alt = float(np.interp(t_mission, gt["timestamp"], gt["alt"]))

    est_err = np.hypot(ee - ge, en - gn)

    # Touchdowns judged from ground truth (real height above the launch point), not PX4's land detector:
    # during a GPS blackout PX4 can sit on the sand believing it is still moving, and never "detect" landing.
    agl = gt_alt - home_alt
    up = agl > 1.0
    lift_idx = np.flatnonzero(~up[:-1] & up[1:]) + 1
    down_idx = np.flatnonzero(up[:-1] & ~up[1:]) + 1
    truth_takeoffs = len(lift_idx)
    sink = np.concatenate([[0.0], -np.diff(gt_alt) * RATE_HZ])  # true descent rate, m/s (positive = down)
    win = int(RATE_HZ)  # look at the last second before contact
    touchdowns = [{"t": sec(tl[i]), "lat": float(gt_lat[i]), "lon": float(gt_lon[i]),
                   "dist_from_home_m": float(math.hypot(ge[i], gn[i])),
                   "speed_ms": float(sink[max(0, i - win):i + 1].max())} for i in down_idx]
    plan_en = list(zip(*_to_en([p[0] for p in meta["plan_latlon"]], [p[1] for p in meta["plan_latlon"]], lat0, lon0)))
    airborne = (gt_alt - home_alt) > 2.0
    deviation = np.where(airborne, _dist_to_polyline(ge, gn, plan_en), 0.0)

    # was the drone flying without a valid height estimate? (that blinds PX4's land detector)
    z_ok = np.interp(tl, lpos["timestamp"], lpos["z_valid"].astype(float)) > 0.5
    height_lost_s = float(np.count_nonzero(airborne & ~z_ok) / RATE_HZ)

    task = sc.task
    coverage = target_miss = None
    if task and task.type == "survey":
        coverage = _coverage_pct(task, ge, gn, agl, airborne)
    elif task and task.type == "delivery":
        tn, te = task.target_ne_m
        target_miss = float(np.hypot(ge[airborne] - te, gn[airborne] - tn).min()) if airborne.any() else None

    # --- fault windows & what happened inside them -----------------------------------------
    windows = _windows(events)
    fault_info = []
    for label, ftype, a, b in windows:
        b = b or t_end
        m = (tl >= a) & (tl <= b + 5e6)  # include 5 s of aftermath
        info = {"label": label, "type": ftype, "start_s": sec(a), "end_s": sec(b),
                "max_estimate_error_m": float(est_err[m].max()) if m.any() else None,
                "max_deviation_m": float(deviation[m].max()) if m.any() else None}
        if ftype == "gps_loss":
            failsafe = next((e for e in events if e["px4_us"] and e["px4_us"] >= a and "ailsafe" in e["text"]), None)
            info["failsafe_after_s"] = (failsafe["px4_us"] - a) / 1e6 if failsafe else None
            info["failsafe_text"] = failsafe["text"] if failsafe else None
            mw = (tl >= a) & (tl <= b)
            info["altitude_lost_m"] = float(gt_alt[mw][0] - gt_alt[mw].min()) if mw.any() else None
        fault_info.append(info)

    # --- landing ----------------------------------------------------------------------------
    landing_dist = touchdowns[-1]["dist_from_home_m"] if touchdowns else float(math.hypot(ge[-1], gn[-1]))
    flight_time = float((tl[down_idx[-1]] - tl[lift_idx[0]]) / 1e6) if truth_takeoffs and len(down_idx) else None
    off_site = [d for d in touchdowns[:-1] if d["dist_from_home_m"] > sc.criteria.max_landing_distance_m]

    c = sc.criteria
    hardest = max((d["speed_ms"] for d in touchdowns), default=0.0)
    optional = [  # (rule, value, limit, unit, passed-if, detail): skipped when the test sets no limit
        ("Position estimate stayed honest", float(est_err.max()), c.max_estimate_error_m, "m", "max",
         "largest gap between where the drone believed it was and where it really was"),
        ("Stayed on the planned route", float(deviation.max()), c.max_path_deviation_m, "m", "max",
         "largest distance of the real drone from the planned route while airborne"),
        ("Landed back at launch", landing_dist, c.max_landing_distance_m, "m", "max",
         "final landing spot vs launch point"),
        ("No hard landing", hardest, c.max_touchdown_speed_ms, "m/s", "max",
         "fastest true descent rate in the last second before any touchdown"),
        ("Covered the search area", coverage, c.min_coverage_pct, "%", "min",
         "share of the area the camera footprint passed over (from ground truth)"),
        ("Reached the drop point", target_miss, c.max_target_miss_m, "m", "max",
         "closest the real drone came to the drop point"),
    ]
    verdicts = [
        Verdict(name, value, limit, unit,
                value is not None and (value <= limit if kind == "max" else value >= limit), detail)
        for name, value, limit, unit, kind, detail in optional if limit is not None
    ] + [
        Verdict("No uncommanded take-off", float(truth_takeoffs), 1.0, "take-offs",
                c.allow_uncommanded_takeoff or truth_takeoffs <= 1,
                f"a second take-off means it touched down and launched again on its own "
                f"(PX4's land detector counted {len(takeoffs_us)})"),
        Verdict("Finished in time", flight_time, c.max_flight_time_s, "s",
                meta["outcome"] == "landed and disarmed" and (flight_time or 1e9) <= c.max_flight_time_s,
                f"run outcome: {meta['outcome']}"),
    ]

    step = max(1, int(RATE_HZ / 2))  # 2.5 Hz is plenty for the report's map and charts
    series = {
        "t": [round(sec(x), 2) for x in tl[::step]],
        "truth_latlon": [[round(a, 7), round(b, 7)] for a, b in zip(gt_lat[::step], gt_lon[::step])],
        "est_latlon": [[round(a, 7), round(b, 7)] for a, b in zip(es_lat[::step], es_lon[::step])],
        "truth_alt": [round(float(x - home_alt), 2) for x in gt_alt[::step]],
        "est_alt": [round(float(x - home_alt), 2) for x in es_alt[::step]],
        "est_err": [round(float(x), 2) for x in est_err[::step]],
        "deviation": [round(float(x), 2) for x in deviation[::step]],
    }
    wind = [{"t": round(sec(w["px4_us"]), 2), "speed": round(math.hypot(w["wind_e"], w["wind_n"]), 2),
             "force": round(math.hypot(w["force_e"], w["force_n"]), 2)}
            for w in meta["wind"] if w["px4_us"] and t_start <= w["px4_us"] <= t_end]

    result = {
        "scenario": {"name": sc.name, "description": sc.description, "location": asdict(sc.location),
                     "air_temp_c": sc.environment.air_temp_c, "wind": asdict(sc.environment.wind),
                     "mission": {"altitude_m": sc.mission.altitude_m, "speed_ms": sc.mission.speed_ms,
                                 "waypoints": len(sc.mission.waypoints_ne_m)}},
        "run": {"outcome": meta["outcome"], "error": meta["error"], "wall_time_s": meta["wall_time_s"],
                "rtf_mean": meta["rtf_mean"], "air_density": meta["air_density"], "dir": run_dir.name},
        "passed": all(v.passed for v in verdicts),
        "verdicts": [asdict(v) for v in verdicts],
        "metrics": {
            "max_estimate_error_m": float(est_err.max()),
            "max_deviation_m": float(deviation.max()),
            "landing_distance_m": landing_dist,
            "takeoffs": truth_takeoffs,
            "px4_detected_takeoffs": int(len(takeoffs_us)),
            "touchdowns": touchdowns,
            "off_site_touchdowns": off_site,
            "flight_time_s": flight_time,
            "max_altitude_m": float((gt_alt - home_alt).max()),
            "hardest_touchdown_ms": hardest,
            "height_estimate_lost_s": height_lost_s,
            "coverage_pct": coverage,
            "target_miss_m": target_miss,
        },
        "faults": fault_info,
        "events": [{"t": sec(e["px4_us"]), "kind": e["kind"], "text": e["text"]}
                   for e in events if e["px4_us"] is not None and e["px4_us"] >= t_start],
        "plan_latlon": meta["plan_latlon"],
        "series": series,
        "wind": wind,
    }
    (run_dir / "analysis.json").write_text(json.dumps(result, indent=1))
    return result
