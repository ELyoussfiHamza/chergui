"""Launch and tear down Gazebo + PX4 SITL for one run, with nothing left behind.

Layout: we start Gazebo ourselves with Chergui's world, then start PX4 in "standalone" mode so it
just spawns the drone into the world we own (instead of starting its own default world).

Every run gets a fresh PX4 working directory, so parameters never leak between runs
(PX4 otherwise remembers params like EKF2_GPS_CTRL in parameters.bson).
"""

from __future__ import annotations

import asyncio
import atexit
import os
import signal
import subprocess
from pathlib import Path

from . import config


class SimError(RuntimeError):
    pass


def _running(pattern: str) -> list[str]:
    out = subprocess.run(["pgrep", "-af", pattern], capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if "pgrep" not in line]


class Simulation:
    def __init__(self, world: str, model: str, run_dir: Path, home: tuple[float, float, float], gui: bool = False,
                 px4_params: dict[str, float] | None = None):
        self.world = world
        self.px4_params = px4_params or {}  # set at boot, so even reboot-required params take effect
        self.home = home  # (lat, lon, alt_m): PX4 moves the world origin here, so the scenario file decides the location
        self.model = model  # PX4 airframe/model name, e.g. "x500"
        self.run_dir = run_dir
        self.gui = gui
        self.px4_dir = run_dir / "px4"
        self._procs: list[asyncio.subprocess.Process] = []

    @property
    def link_name(self) -> str:
        return f"{self.model}_0::base_link"  # PX4 instance 0 spawns the model as "<model>_0"

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        path = lambda *p: os.pathsep.join(str(x) for x in p)  # noqa: E731
        env.update(
            GZ_SIM_RESOURCE_PATH=path(config.PX4_GZ_MODELS, config.WORLDS_DIR),
            GZ_SIM_SYSTEM_PLUGIN_PATH=str(config.PX4_GZ_PLUGINS),
            GZ_SIM_SERVER_CONFIG_PATH=str(config.PX4_GZ_SERVER_CONFIG),
            GZ_IP="127.0.0.1",
            PX4_GZ_STANDALONE="1",
            PX4_GZ_WORLD=self.world,
            PX4_GZ_MODELS=str(config.PX4_GZ_MODELS),
            PX4_SIM_MODEL=f"gz_{self.model}",
            HEADLESS="1",
            PX4_HOME_LAT=str(self.home[0]),
            PX4_HOME_LON=str(self.home[1]),
            PX4_HOME_ALT=str(self.home[2]),
        )
        # PX4's startup script applies every PX4_PARAM_<name>=<value> before any module starts.
        env.update({f"PX4_PARAM_{k}": str(v) for k, v in self.px4_params.items()})
        return env

    def preflight(self) -> None:
        if not config.PX4_BIN.exists():
            raise SimError(f"PX4 SITL binary not found at {config.PX4_BIN}. Build it with: make px4_sitl")
        busy = _running(r"bin/px4( |$)") + _running(r"gz sim")
        if busy:
            raise SimError("another PX4 or Gazebo is already running (ports would clash):\n  "
                           + "\n  ".join(b[:120] for b in busy)
                           + "\nStop it first (pxh> shutdown, or close the terminal running make px4_sitl).")

    async def _spawn(self, name: str, *cmd: str, cwd: Path | None = None) -> asyncio.subprocess.Process:
        log = open(self.run_dir / f"{name}.log", "wb")
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=cwd, env=self._env(), stdin=asyncio.subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,  # own process group → clean kill
        )
        self._procs.append(proc)
        if len(self._procs) == 1:
            atexit.register(self._kill_leftovers)
        return proc

    def _kill_leftovers(self) -> None:
        """Safety net: a second Ctrl+C can abort the async teardown, and the simulators run in their own
        process groups, so nothing else would stop them. Runs at interpreter exit; a no-op after a clean stop."""
        for proc in self._procs:
            if proc.returncode is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass

    async def start(self) -> None:
        self.preflight()
        self.px4_dir.mkdir(parents=True, exist_ok=True)
        world_file = config.WORLDS_DIR / f"{self.world}.sdf"
        await self._spawn("gazebo", "gz", "sim", "-s", "-r", "--headless-rendering", str(world_file))
        if self.gui:
            await self._spawn("gazebo_gui", "gz", "sim", "-g")
        # -d: daemon mode (no interactive pxh> shell, which would spin on a closed stdin)
        # first arg: PX4's etc/ dir (startup scripts), symlinked into -w, the per-run working dir (params, logs)
        await self._spawn("px4", str(config.PX4_BIN), "-d", str(config.PX4_BUILD / "etc"), "-w", str(self.px4_dir))

    def check_alive(self) -> None:
        for p in self._procs:
            if p.returncode is not None:
                raise SimError(f"a simulator process exited early (code {p.returncode}); see logs in {self.run_dir}")

    async def stop(self) -> None:
        # PX4 first, with SIGINT so it shuts down cleanly and closes the flight log properly.
        for proc in reversed(self._procs):
            if proc.returncode is not None:
                continue
            try:
                os.killpg(proc.pid, signal.SIGINT)
                await asyncio.wait_for(proc.wait(), timeout=10)
            except (ProcessLookupError, asyncio.TimeoutError):
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                    await proc.wait()
                except ProcessLookupError:
                    pass

    def ulog_path(self) -> Path | None:
        logs = sorted((self.px4_dir / "log").glob("**/*.ulg"))
        return logs[-1] if logs else None
