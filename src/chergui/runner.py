"""One run = one scenario flown once: launch sim → fly mission → inject faults → land → collect evidence."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from . import config
from .scenario import Fault, Scenario
from .sim import Simulation
from .vehicle import Vehicle
from .wind import WindController, air_density

log = logging.getLogger("chergui.run")

WORLD = "sahara"


async def _fault_trigger(fault: Fault, vehicle: Vehicle, mission_start: float) -> None:
    """Wait until the fault's trigger condition is met."""
    if fault.at_s is not None:
        await asyncio.sleep(max(0.0, mission_start + fault.at_s - time.monotonic()))
    elif fault.at_progress is not None:  # independent of how many waypoints the plan has
        while vehicle.progress[1] == 0 or vehicle.progress[0] / vehicle.progress[1] < fault.at_progress:
            await asyncio.sleep(0.2)
    else:
        while vehicle.progress[0] < fault.at_waypoint:
            await asyncio.sleep(0.2)
    await asyncio.sleep(fault.delay_s)


async def _run_fault(fault: Fault, vehicle: Vehicle, wind: WindController, mission_start: float) -> None:
    await _fault_trigger(fault, vehicle, mission_start)
    if fault.type == "gust":
        wind.gust(fault.duration_s, fault.speed_ms, fault.from_deg)
        vehicle.event("fault", f"gust {fault.speed_ms:.0f} m/s from {fault.from_deg:.0f}° for {fault.duration_s:.0f} s")
        await asyncio.sleep(fault.duration_s)
        vehicle.event("fault_end", "gust over")
    elif fault.type == "gps_loss":
        await vehicle.set_gps(False)
        vehicle.event("fault", f"GNSS blackout for {fault.duration_s:.0f} s")
        await asyncio.sleep(fault.duration_s)
        await vehicle.set_gps(True)
        vehicle.event("fault_end", "GNSS restored")


async def _wait_for_end(vehicle: Vehicle, sim: Simulation, timeout_s: float) -> str:
    """The flight is over when the drone has flown and is disarmed again, or on timeout."""
    deadline = time.monotonic() + timeout_s
    flown = False
    while time.monotonic() < deadline:
        sim.check_alive()
        flown = flown or vehicle.in_air
        if flown and not vehicle.armed:
            return "landed and disarmed"
        await asyncio.sleep(0.5)
    return "timeout"


async def run(scenario: Scenario, gui: bool = False, run_dir: Path | None = None) -> Path:
    if run_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = config.RUNS_DIR / f"{stamp}_{scenario.slug}"
    run_dir.mkdir(parents=True)
    if scenario.source:
        shutil.copy(scenario.source, run_dir / "scenario.yaml")

    loc = scenario.location
    sim = Simulation(WORLD, scenario.vehicle, run_dir, home=(loc.lat, loc.lon, loc.alt_m), gui=gui,
                     px4_params=scenario.px4_params)
    vehicle = Vehicle(config.MAVLINK_URL)
    env = scenario.environment
    wind = WindController(
        world=WORLD, link=sim.link_name, base_speed_ms=env.wind.speed_ms, base_from_deg=env.wind.from_deg,
        rho=air_density(env.air_temp_c, scenario.location.alt_m), drag_area_m2=scenario.drag_area_m2,
        clock=vehicle.clock,
    )
    outcome, plan, error = "not started", [], None
    started = time.monotonic()
    try:
        log.info("starting Gazebo + PX4 (run dir: %s)", run_dir)
        await sim.start()
        await vehicle.connect()
        vehicle.start_monitors()
        await vehicle.wait_until_ready()
        await wind.start()
        vehicle.event("env", f"wind {env.wind.speed_ms:.0f} m/s from {env.wind.from_deg:.0f}°, "
                             f"air {env.air_temp_c:.0f} °C (ρ={wind.rho:.3f} kg/m³)")
        m = scenario.mission
        plan = await vehicle.upload_mission(m.waypoints_ne_m, m.altitude_m, m.speed_ms, m.return_home)
        await vehicle.start_mission()
        mission_start = time.monotonic()
        faults = [asyncio.create_task(_run_fault(f, vehicle, wind, mission_start)) for f in scenario.faults]
        outcome = await _wait_for_end(vehicle, sim, scenario.criteria.max_flight_time_s)
        for f in faults:
            f.cancel()
        vehicle.event("run", f"flight ended: {outcome}")
    except Exception as exc:  # keep the evidence even when a run goes wrong
        error = f"{type(exc).__name__}: {exc}"
        outcome = "error"
        log.error("run failed: %s", error)
    finally:
        await wind.stop()
        await vehicle.close()
        await sim.stop()

    ulog = sim.ulog_path()
    if ulog:
        shutil.copy(ulog, run_dir / "flight.ulg")
    meta = {
        "scenario": scenario.name,
        "outcome": outcome,
        "error": error,
        "wall_time_s": round(time.monotonic() - started, 1),
        "rtf_mean": round(sum(wind.rtf) / len(wind.rtf), 3) if wind.rtf else None,
        "air_density": wind.rho,
        "plan_latlon": plan,
        "events": [asdict(e) for e in vehicle.events],
        "wind": [asdict(s) for s in wind.samples[::5]],  # 4 Hz is plenty for plots
    }
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2))
    return run_dir
