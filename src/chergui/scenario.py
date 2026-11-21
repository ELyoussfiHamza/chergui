"""Scenario files: one YAML file fully describes an experiment, so any run can be repeated exactly."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


class ScenarioError(ValueError):
    pass


@dataclass
class Location:
    name: str
    lat: float
    lon: float
    alt_m: float


@dataclass
class Wind:
    speed_ms: float = 0.0
    from_deg: float = 90.0  # meteorological: where the wind comes FROM. Chergui blows from the east.


@dataclass
class Environment:
    air_temp_c: float = 25.0
    wind: Wind = field(default_factory=Wind)


@dataclass
class Mission:
    altitude_m: float
    speed_ms: float
    waypoints_ne_m: list[tuple[float, float]]  # (north, east) offsets from home, metres
    return_home: bool = True


@dataclass
class Task:
    """What the mission is FOR. Part of the test: the planner may not change it."""

    type: str  # "survey" (cover an area) | "delivery" (reach a point and come back)
    area_center_ne_m: tuple[float, float] | None = None  # survey: rectangle centre, metres north/east of launch
    area_size_m: tuple[float, float] | None = None  # survey: (north-south length, east-west width)
    target_ne_m: tuple[float, float] | None = None  # delivery: drop point, metres north/east of launch
    camera_hfov_deg: float = 70.0  # survey: downward camera, sets the ground footprint per altitude


@dataclass
class Fault:
    type: str  # "gust" | "gps_loss"
    duration_s: float
    at_s: float | None = None  # seconds after mission start ...
    at_waypoint: int | None = None  # ... or once waypoint N (1-based) has been reached ...
    at_progress: float | None = None  # ... or once this fraction (0-1) of the waypoints has been reached
    delay_s: float = 0.0  # extra wait after the trigger
    # gust only
    speed_ms: float = 0.0
    from_deg: float = 90.0


@dataclass
class Criteria:
    """Pass/fail limits. A limit set to null in the YAML turns that rule into a measurement only."""

    max_estimate_error_m: float | None = 15.0
    max_path_deviation_m: float | None = 30.0
    max_landing_distance_m: float | None = 10.0
    max_flight_time_s: float = 400.0  # always judged: it is also the run's timeout
    allow_uncommanded_takeoff: bool = False
    max_touchdown_speed_ms: float | None = None  # hard-landing rule
    min_coverage_pct: float | None = None  # survey task
    max_target_miss_m: float | None = None  # delivery task


@dataclass
class Scenario:
    name: str
    description: str
    location: Location
    vehicle: str
    drag_area_m2: float
    environment: Environment
    mission: Mission
    faults: list[Fault]
    criteria: Criteria
    task: Task | None = None
    px4_params: dict[str, float] = field(default_factory=dict)  # PX4 settings applied at boot
    source: Path | None = None

    @property
    def slug(self) -> str:
        return self.source.stem if self.source else "scenario"


FAULT_TYPES = {"gust", "gps_loss"}
TASK_TYPES = {"survey", "delivery"}
TRIGGERS = ("at_s", "at_waypoint", "at_progress")


def parse_task(t: dict, where: str = "task") -> Task:
    if t.get("type") not in TASK_TYPES:
        raise ScenarioError(f"{where}: task.type must be one of {sorted(TASK_TYPES)}")
    try:
        task = Task(**{k: tuple(v) if isinstance(v, list) else v for k, v in t.items()})
    except TypeError as exc:
        raise ScenarioError(f"{where}: bad task fields ({exc})") from None
    if task.type == "survey" and not (task.area_center_ne_m and task.area_size_m):
        raise ScenarioError(f"{where}: a survey task needs area_center_ne_m and area_size_m")
    if task.type == "delivery" and not task.target_ne_m:
        raise ScenarioError(f"{where}: a delivery task needs target_ne_m")
    return task


def load(path: str | Path) -> Scenario:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except FileNotFoundError:
        raise ScenarioError(f"scenario file not found: {path}") from None

    def need(d: dict, key: str, where: str):
        if key not in d:
            raise ScenarioError(f"{path.name}: missing '{key}' in {where}")
        return d[key]

    loc = need(raw, "location", "top level")
    env = raw.get("environment", {})
    wind = env.get("wind", {})
    mis = need(raw, "mission", "top level")

    wps = need(mis, "waypoints_ne_m", "mission")
    if not wps or any(len(w) != 2 for w in wps):
        raise ScenarioError(f"{path.name}: mission.waypoints_ne_m must be a list of [north, east] pairs")

    faults = []
    for i, f in enumerate(raw.get("faults", []), 1):
        ftype = need(f, "type", f"fault #{i}")
        if ftype not in FAULT_TYPES:
            raise ScenarioError(f"{path.name}: fault #{i} has unknown type '{ftype}' (known: {sorted(FAULT_TYPES)})")
        if sum(k in f for k in TRIGGERS) != 1:
            raise ScenarioError(f"{path.name}: fault #{i} needs exactly one of {', '.join(TRIGGERS)}")
        if "at_waypoint" in f and not 1 <= f["at_waypoint"] <= len(wps):
            raise ScenarioError(f"{path.name}: fault #{i} at_waypoint must be between 1 and {len(wps)}")
        if "at_progress" in f and not 0 < f["at_progress"] < 1:
            raise ScenarioError(f"{path.name}: fault #{i} at_progress must be between 0 and 1")
        faults.append(Fault(**f))

    task = parse_task(raw["task"], path.name) if raw.get("task") else None

    return Scenario(
        name=need(raw, "name", "top level"),
        description=raw.get("description", ""),
        location=Location(**loc),
        vehicle=raw.get("vehicle", "x500"),
        drag_area_m2=raw.get("drag_area_m2", 0.08),
        environment=Environment(air_temp_c=env.get("air_temp_c", 25.0), wind=Wind(**wind)),
        mission=Mission(
            altitude_m=need(mis, "altitude_m", "mission"),
            speed_ms=need(mis, "speed_ms", "mission"),
            waypoints_ne_m=[tuple(w) for w in wps],
            return_home=mis.get("return_home", True),
        ),
        faults=faults,
        criteria=Criteria(**raw.get("criteria", {})),
        task=task,
        px4_params=dict(raw.get("px4_params") or {}),
        source=path,
    )
