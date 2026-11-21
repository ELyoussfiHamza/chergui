"""Wind model: turns "6 m/s Chergui with a 15 m/s gust" into a force on the drone.

Physics, kept deliberately simple and explainable:
  drag force  F = ½ · ρ · CdA · |v| · v          (v = wind relative to the drone, CdA = drag area)
  air density ρ = p / (R · T)                     (hot, high air is thinner, so it pushes less)
  gust shape  1-cosine ramp, the standard discrete-gust profile used in aviation

The wind is computed here; the force is computed in the bridge, which knows the drone's true velocity
(v = wind − drone velocity). A drifting drone therefore tops out at wind speed, like a real one.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass, field

from . import config

R_AIR = 287.05  # J/(kg·K)


def air_density(temp_c: float, alt_m: float) -> float:
    """ISA pressure at altitude, actual temperature: e.g. 45 °C at 715 m → ~1.03 kg/m³ (vs 1.225 at sea level, 15 °C)."""
    pressure = 101325.0 * (1 - 2.25577e-5 * alt_m) ** 5.25588
    return pressure / (R_AIR * (temp_c + 273.15))


def wind_vector(speed_ms: float, from_deg: float) -> tuple[float, float]:
    """Meteorological direction → ENU vector the air moves along (x=east, y=north)."""
    toward = math.radians(from_deg + 180.0)
    return speed_ms * math.sin(toward), speed_ms * math.cos(toward)


def one_minus_cosine(t: float, duration: float) -> float:
    """0 → 1 → 0 over `duration` seconds, smooth at both ends."""
    if t <= 0 or t >= duration:
        return 0.0
    return 0.5 * (1 - math.cos(2 * math.pi * t / duration))


@dataclass
class Gust:
    start: float  # monotonic seconds
    duration_s: float
    speed_ms: float
    from_deg: float


@dataclass
class WindSample:
    t: float  # monotonic seconds
    px4_us: int | None
    wind_e: float
    wind_n: float
    force_e: float
    force_n: float


@dataclass
class WindController:
    """Owns the wind bridge process and streams forces to it at a fixed rate."""

    world: str
    link: str
    base_speed_ms: float
    base_from_deg: float
    rho: float
    drag_area_m2: float
    clock: object = None  # anything with .px4_us (set by the runner for log alignment)
    rate_hz: float = 20.0
    gusts: list[Gust] = field(default_factory=list)
    samples: list[WindSample] = field(default_factory=list)
    rtf: list[float] = field(default_factory=list)
    last_force: tuple[float, float] = (0.0, 0.0)  # as reported by the bridge

    async def start(self) -> None:
        bridge = config.PROJECT_ROOT / "src" / "chergui" / "bridge" / "wind_bridge.py"
        self._proc = await asyncio.create_subprocess_exec(
            config.SYSTEM_PYTHON, str(bridge), self.world, self.link,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        async def _handshake():
            async for line in self._proc.stdout:  # stats lines may arrive before "ready"
                if json.loads(line).get("ready"):
                    return
            raise RuntimeError("wind bridge exited during startup")

        await asyncio.wait_for(_handshake(), timeout=20)
        self._reader = asyncio.create_task(self._read_stats())
        self._loop = asyncio.create_task(self._run())

    def gust(self, duration_s: float, speed_ms: float, from_deg: float) -> None:
        self.gusts.append(Gust(time.monotonic(), duration_s, speed_ms, from_deg))

    def current_wind(self, now: float) -> tuple[float, float]:
        we, wn = wind_vector(self.base_speed_ms, self.base_from_deg)
        for g in self.gusts:
            k = one_minus_cosine(now - g.start, g.duration_s)
            if k:
                ge, gn = wind_vector(g.speed_ms - self.base_speed_ms, g.from_deg)
                we, wn = we + k * ge, wn + k * gn
        return we, wn

    @property
    def k(self) -> float:
        return 0.5 * self.rho * self.drag_area_m2

    async def _run(self) -> None:
        period = 1.0 / self.rate_hz
        while True:
            now = time.monotonic()
            we, wn = self.current_wind(now)
            fe, fn = self.last_force
            self.samples.append(WindSample(now, getattr(self.clock, "px4_us", None), we, wn, fe, fn))
            self._proc.stdin.write((json.dumps({"wind": [we, wn], "k": self.k}) + "\n").encode())
            await self._proc.stdin.drain()
            await asyncio.sleep(period)

    async def _read_stats(self) -> None:
        async for line in self._proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if "rtf" in msg:
                self.rtf.append(float(msg["rtf"]))
            elif "force" in msg:
                self.last_force = tuple(msg["force"])

    async def stop(self) -> None:
        for task in (getattr(self, "_loop", None), getattr(self, "_reader", None)):
            if task:
                task.cancel()
        proc = getattr(self, "_proc", None)
        if proc and proc.returncode is None:
            proc.stdin.close()  # bridge clears the force and exits
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                proc.kill()
