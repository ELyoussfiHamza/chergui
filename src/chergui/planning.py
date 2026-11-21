"""Turn a plan (a few numbers the AI chooses) into waypoints, and check it before anything flies.

Split of responsibilities:
  the TEST  (human-approved, locked): place, weather, faults, the task (area / drop point), pass/fail rules
  the PLAN  (AI-written):             altitude, speed, pattern choices, a short allowlist of PX4 settings
The AI never writes coordinates. Geometry is computed here, so a plan can't put a waypoint somewhere odd.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from .scenario import Task

# PX4 settings the planner may change. Rule: safety settings can only be made STRICTER than PX4's defaults.
# (First version allowed the full PX4 ranges, and the planner's "fix" for a GNSS blackout was to delay and
# loosen the position-loss failsafe so the drone kept flying blind. It drifted 188 m. Never again.)
PX4_ALLOWLIST: dict[str, dict] = {
    "EKF2_HGT_REF": {"min": 0, "max": 1, "int": True,
                     "doc": "main height source for the estimator: 0 = barometer, 1 = GNSS (PX4 default 1)"},
    "EKF2_NOAID_TOUT": {"min": 1_000_000, "max": 5_000_000, "int": True,
                        "doc": "microseconds the estimator may dead-reckon without position aiding before it "
                               "declares position invalid (default 5000000; may only be shortened)"},
    "COM_POS_FS_EPH": {"min": 1, "max": 5,
                       "doc": "horizontal position uncertainty (m) that triggers the position-loss failsafe "
                              "(default 5; may only be lowered)"},
    "COM_FAIL_ACT_T": {"min": 0, "max": 5,
                       "doc": "seconds the drone holds before executing a failsafe action (default 5; may only be shortened)"},
    "MPC_LAND_SPEED": {"min": 0.6, "max": 1.2, "doc": "descent speed in m/s for the final landing phase (default 0.7)"},
}

LIMITS = {
    "altitude_m": (10.0, 60.0),
    "speed_ms": (2.0, 12.0),
    "line_spacing_m": (5.0, 80.0),
}
CLIMB_MS = 1.5  # rough numbers for the flight-time estimate
LAND_MS = 0.8
TIME_MARGIN = 0.8  # the estimated flight must fit in 80% of the test's time limit


@dataclass
class Plan:
    altitude_m: float
    speed_ms: float
    line_spacing_m: float | None = None  # survey only
    line_direction: str = "north-south"  # survey only: direction of the long passes
    px4_params: dict[str, float] = field(default_factory=dict)
    rationale: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def footprint_m(altitude_m: float, hfov_deg: float) -> float:
    """Width of ground a downward camera sees at this height."""
    return 2 * altitude_m * math.tan(math.radians(hfov_deg / 2))


def waypoints(task: Task, plan: Plan) -> list[tuple[float, float]]:
    """(north, east) offsets from launch, metres."""
    if task.type == "delivery":
        return [tuple(task.target_ne_m)]

    # survey: a lawnmower pattern over the rectangle, passes inset half a spacing from the edges
    (cn, ce), (length, width) = task.area_center_ne_m, task.area_size_m
    s = plan.line_spacing_m
    ns = plan.line_direction == "north-south"
    across, along = (width, length) if ns else (length, width)
    n_lines = max(1, math.ceil(across / s))
    offsets = [-across / 2 + across * (i + 0.5) / n_lines for i in range(n_lines)]
    pts = []
    for i, off in enumerate(offsets):
        ends = (-along / 2, along / 2) if i % 2 == 0 else (along / 2, -along / 2)
        for a in ends:
            pts.append((cn + a, ce + off) if ns else (cn + off, ce + a))
    return pts


def route_length_m(pts: list[tuple[float, float]], return_home: bool = True) -> float:
    path = [(0.0, 0.0)] + list(pts) + ([(0.0, 0.0)] if return_home else [])
    return sum(math.dist(a, b) for a, b in zip(path, path[1:]))


def estimate_flight_s(pts, plan: Plan) -> float:
    return plan.altitude_m / CLIMB_MS + route_length_m(pts) / plan.speed_ms + plan.altitude_m / LAND_MS


def check(task: Task, max_flight_time_s: float, plan: Plan) -> list[str]:
    """Deterministic pre-flight checks. An empty list means the plan may fly."""
    errors = []
    for name, (lo, hi) in LIMITS.items():
        v = getattr(plan, name)
        if v is None:
            continue
        if not lo <= v <= hi:
            errors.append(f"{name}={v} is outside the allowed range {lo}-{hi}")
    for k, v in plan.px4_params.items():
        spec = PX4_ALLOWLIST.get(k)
        if spec is None:
            errors.append(f"PX4 setting {k} is not on the allowlist ({', '.join(PX4_ALLOWLIST)})")
        elif not spec["min"] <= v <= spec["max"]:
            errors.append(f"{k}={v} is outside its allowed range {spec['min']}-{spec['max']}")
        elif spec.get("int") and v != int(v):
            errors.append(f"{k} must be a whole number")

    if task.type == "survey":
        if plan.line_spacing_m is None:
            errors.append("a survey plan needs line_spacing_m")
        elif plan.line_direction not in ("north-south", "east-west"):
            errors.append("line_direction must be 'north-south' or 'east-west'")
        else:
            fp = footprint_m(plan.altitude_m, task.camera_hfov_deg)
            if plan.line_spacing_m > fp:
                errors.append(f"line spacing {plan.line_spacing_m:.0f} m is wider than the camera footprint at "
                              f"{plan.altitude_m:.0f} m ({fp:.0f} m): the survey would leave gaps")
    if not errors:
        pts = waypoints(task, plan)
        est = estimate_flight_s(pts, plan)
        budget = max_flight_time_s * TIME_MARGIN
        if est > budget:
            errors.append(f"estimated flight time {est:.0f} s exceeds {budget:.0f} s "
                          f"(80% of the {max_flight_time_s:.0f} s limit)")
        if len(pts) > 60:
            errors.append(f"{len(pts)} waypoints is too many (max 60): widen the line spacing")
    return errors
