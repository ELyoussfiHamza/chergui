"""Everything that talks to the drone over MAVLink (via MAVSDK)."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field

from mavsdk.asyncio import ComponentType, Configuration, Mavsdk
from mavsdk.asyncio.plugins.action import ActionAsync
from mavsdk.asyncio.plugins.mission import MissionAsync, MissionItem, MissionPlan
from mavsdk.asyncio.plugins.param import ParamAsync
from mavsdk.asyncio.plugins.action import ActionError
from mavsdk.asyncio.plugins.telemetry import TelemetryAsync
from mavsdk.logging import LogLevel, log_subscribe
from mavsdk.plugins.telemetry.telemetry import FlightMode

log = logging.getLogger("chergui.vehicle")


def _quiet_mavsdk(level, message, file, line) -> bool:
    """Route MAVSDK's own logging: drop debug chatter, keep warnings. Returning True = 'handled, don't print'."""
    if level >= LogLevel.WARN:
        log.warning("mavsdk: %s", message)
    return True


log_subscribe(_quiet_mavsdk)

EARTH_R = 6378137.0


def offset_to_latlon(lat0: float, lon0: float, north_m: float, east_m: float) -> tuple[float, float]:
    """Flat-earth offset, accurate to centimetres over a few hundred metres."""
    lat = lat0 + math.degrees(north_m / EARTH_R)
    lon = lon0 + math.degrees(east_m / (EARTH_R * math.cos(math.radians(lat0))))
    return lat, lon


@dataclass
class Event:
    t: float  # monotonic seconds
    px4_us: int | None  # PX4 boot clock, same clock as the flight log → lets us place events on log plots
    kind: str
    text: str


@dataclass
class Px4Clock:
    """Latest PX4 timestamp seen on the IMU stream. PX4 runs lockstep with Gazebo, so this is sim time."""

    px4_us: int | None = None


@dataclass
class Vehicle:
    url: str
    clock: Px4Clock = field(default_factory=Px4Clock)
    events: list[Event] = field(default_factory=list)
    armed: bool = False
    in_air: bool = False
    flight_mode: str = "?"
    progress: tuple[int, int] = (0, 0)

    def event(self, kind: str, text: str) -> Event:
        e = Event(time.monotonic(), self.clock.px4_us, kind, text)
        self.events.append(e)
        log.info("[%s] %s", kind, text)
        return e

    async def connect(self, timeout_s: float = 60) -> None:
        self._mavsdk = Mavsdk(Configuration.create_with_component_type(ComponentType.GROUND_STATION))
        await self._mavsdk.add_any_connection(self.url)
        drone = await self._mavsdk.first_autopilot(timeout_s=timeout_s)
        if drone is None:
            raise TimeoutError(f"no autopilot on {self.url} after {timeout_s:.0f} s")
        self.drone = drone
        self.action = ActionAsync(drone)
        self.mission = MissionAsync(drone)
        self.param = ParamAsync(drone)
        self.telemetry = TelemetryAsync(drone)
        self.event("link", "connected to PX4")

    async def wait_until_ready(self, timeout_s: float = 120) -> None:
        """Block until PX4 has a position fix and a home position (your step-2 health check)."""

        stable_for = 3.0  # PX4 can report OK for a moment while the estimator is still settling

        async def _wait():
            good_since = None
            async for h in self.telemetry.subscribe_health():
                ok = h.is_global_position_ok and h.is_home_position_ok and h.is_armable
                if not ok:
                    good_since = None
                elif good_since is None:
                    good_since = time.monotonic()
                elif time.monotonic() - good_since >= stable_for:
                    return

        await asyncio.wait_for(_wait(), timeout_s)
        self.event("health", f"ready to fly (position + home OK for {stable_for:.0f} s)")

    def start_monitors(self) -> None:
        async def clock():
            await self.telemetry.set_rate_imu(50.0)
            async for imu in self.telemetry.subscribe_imu():
                self.clock.px4_us = imu.timestamp_us

        async def status():
            async for s in self.telemetry.subscribe_status_text():
                self.event("px4", s.text)

        async def armed():
            async for a in self.telemetry.subscribe_armed():
                if a != self.armed:
                    self.armed = a
                    self.event("state", "armed" if a else "disarmed")

        async def in_air():
            async for a in self.telemetry.subscribe_in_air():
                if a != self.in_air:
                    self.in_air = a
                    self.event("state", "airborne" if a else "on the ground")

        async def mode():
            async for m in self.telemetry.subscribe_flight_mode():
                name = FlightMode(m).name
                if name != self.flight_mode:
                    self.flight_mode = name
                    self.event("mode", name)

        async def progress():
            async for p in self.mission.subscribe_mission_progress():
                if (p.current, p.total) != self.progress:
                    self.progress = (p.current, p.total)
                    self.event("mission", f"waypoint {p.current}/{p.total}")

        self._monitors = [asyncio.create_task(c()) for c in (clock, status, armed, in_air, mode, progress)]

    async def home(self) -> tuple[float, float]:
        h = await self.telemetry.home()
        return h.latitude_deg, h.longitude_deg

    async def upload_mission(self, waypoints_ne_m, altitude_m: float, speed_ms: float, return_home: bool) -> list:
        lat0, lon0 = await self.home()
        items, plan_ll = [], []
        for north, east in waypoints_ne_m:
            lat, lon = offset_to_latlon(lat0, lon0, north, east)
            plan_ll.append((lat, lon))
            items.append(MissionItem(latitude_deg=lat, longitude_deg=lon, relative_altitude_m=altitude_m,
                                     speed_m_s=speed_ms, is_fly_through=True, acceptance_radius_m=3.0))
        await self.mission.set_return_to_launch_after_mission(return_home)
        await self.mission.upload_mission(MissionPlan(items))
        self.event("mission", f"uploaded {len(items)} waypoints at {altitude_m:.0f} m")
        return [(lat0, lon0)] + plan_ll + ([(lat0, lon0)] if return_home else [])

    async def start_mission(self, attempts: int = 10) -> None:
        for i in range(attempts):  # PX4 may still deny arming for a few seconds after boot
            try:
                await self.action.arm()
                break
            except ActionError as exc:
                if i == attempts - 1:
                    raise
                self.event("health", f"arming denied ({exc}), retrying")
                await asyncio.sleep(2)
        await self.mission.start_mission()
        self.event("mission", "mission started")

    async def set_gps(self, enabled: bool) -> None:
        # EKF2_GPS_CTRL bitmask: 1 lat/lon + 2 altitude + 4 velocity = 7 (all). 0 = estimator ignores GNSS.
        # (PX4's `failure gps off` is not implemented for Gazebo Harmonic, so we cut GNSS at the estimator.)
        await self.param.set_param_int("EKF2_GPS_CTRL", 7 if enabled else 0)

    async def close(self) -> None:
        for t in getattr(self, "_monitors", []):
            t.cancel()
        if hasattr(self, "_mavsdk"):
            self._mavsdk.destroy()
