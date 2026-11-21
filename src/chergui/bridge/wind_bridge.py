#!/usr/bin/python3
"""Wind bridge: pushes wind forces onto the drone inside Gazebo.

Why a separate process: Gazebo's Python bindings (gz.transport13 / gz.msgs10) are packaged for
the system Python 3.10, while Chergui runs on Python 3.13 in its own venv. The `gz topic` CLI
would work from anywhere, but each call takes ~3 s, far too slow to shape a gust.
So this tiny script runs under /usr/bin/python3 and talks to the main process over stdin/stdout:

  stdin  (one JSON per line):  {"wind": [we, wn], "k": k}   wind m/s (x=east, y=north), k = ½·ρ·CdA
  stdout (one JSON per line):  {"ready": true} once, then {"sim_time": s, "rtf": r} about once a second,
                               and {"force": [fe, fn], "vel": [ve, vn]} a few times a second

Drag uses the wind RELATIVE TO THE DRONE: F = k·|w − v|·(w − v), with v the drone's true velocity,
differentiated from Gazebo's pose stream (the same /pose/info topic PX4 logs as ground truth).
Without that, the push never weakens as the drone speeds up downwind, and a drifting drone would
accelerate past the wind speed, which is physically impossible. That bug was real; it inflated drift.

Forces go through Gazebo's ApplyLinkWrench system as a *persistent* wrench on the drone's base
link: it keeps being applied every physics step.

Gotcha (measured, see README): persistent wrenches ADD UP. Each message is appended to the
link's list; it does not replace the previous one. And a "clear" sent in the same instant as a
new wrench may be processed after it, wiping the new one. So we send only the *change* in force,
and every RESYNC_S we clear and re-apply the full force, so a lost message can't leave error behind.

Standard library + Gazebo bindings only. Do not import anything from the chergui package here.
"""

import json
import sys
import time

RESYNC_S = 2.0

from gz.msgs10.entity_pb2 import Entity
from gz.msgs10.entity_wrench_pb2 import EntityWrench
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.world_stats_pb2 import WorldStatistics
from gz.transport13 import Node


def main() -> None:
    world, link = sys.argv[1], sys.argv[2]  # e.g. "sahara", "x500_0::base_link"
    node = Node()
    pub_set = node.advertise(f"/world/{world}/wrench/persistent", EntityWrench)
    pub_clear = node.advertise(f"/world/{world}/wrench/clear", Entity)

    target = Entity()
    target.name = link
    target.type = Entity.LINK

    last_report = [0.0]

    def on_stats(msg: WorldStatistics) -> None:
        now = time.monotonic()
        if now - last_report[0] >= 1.0:
            last_report[0] = now
            sim_t = msg.sim_time.sec + msg.sim_time.nsec * 1e-9
            print(json.dumps({"sim_time": sim_t, "rtf": msg.real_time_factor}), flush=True)

    node.subscribe(WorldStatistics, f"/world/{world}/stats", on_stats)

    # true velocity of the drone, from consecutive poses (sim-time stamped), lightly low-passed
    model = link.split("::")[0]
    state = {"t": None, "x": 0.0, "y": 0.0, "v": [0.0, 0.0]}
    SMOOTH = 0.3  # per pose update (~250 Hz): ~15 ms time constant, removes finite-difference jitter

    def on_pose(msg: Pose_V) -> None:
        for p in msg.pose:
            if p.name != model:
                continue
            t = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
            if state["t"] is not None and t > state["t"]:
                dt = t - state["t"]
                ve, vn = (p.position.x - state["x"]) / dt, (p.position.y - state["y"]) / dt
                v = state["v"]
                state["v"] = [v[0] + SMOOTH * (ve - v[0]), v[1] + SMOOTH * (vn - v[1])]
            state["t"], state["x"], state["y"] = t, p.position.x, p.position.y
            return

    node.subscribe(Pose_V, f"/world/{world}/pose/info", on_pose)

    # Publishers need a moment to be discovered before the first message is delivered.
    time.sleep(1.0)
    print(json.dumps({"ready": True}), flush=True)

    def push(f) -> None:
        w = EntityWrench()
        w.entity.CopyFrom(target)
        w.wrench.force.x, w.wrench.force.y, w.wrench.force.z = f
        pub_set.publish(w)

    applied = [0.0, 0.0, 0.0]  # what Gazebo currently applies, by our bookkeeping
    last_resync = time.monotonic()
    last_echo = 0.0
    # readline() instead of `for line in sys.stdin`: the iterator reads ahead in 8 KB chunks,
    # which would deliver our 20 Hz updates in bursts every few seconds.
    for line in iter(sys.stdin.readline, ""):
        try:
            cmd = json.loads(line)
            we, wn = (float(x) for x in cmd["wind"])
            k = float(cmd["k"])
        except (ValueError, KeyError, TypeError):
            continue
        ve, vn = state["v"]
        re_, rn = we - ve, wn - vn  # air velocity as the drone feels it
        rel = (re_ * re_ + rn * rn) ** 0.5
        want = [k * rel * re_, k * rel * rn, 0.0]
        now = time.monotonic()
        if now - last_echo >= 0.25:
            last_echo = now
            print(json.dumps({"force": [round(want[0], 3), round(want[1], 3)], "vel": [round(ve, 2), round(vn, 2)]}),
                  flush=True)
        if now - last_resync >= RESYNC_S:
            pub_clear.publish(target)
            time.sleep(0.03)  # let the clear land several physics steps (4 ms each) before re-applying
            applied = [0.0, 0.0, 0.0]
            last_resync = now
        delta = [w - a for w, a in zip(want, applied)]
        if max(abs(d) for d in delta) > 0.01:
            push(delta)
            applied = want

    pub_clear.publish(target)  # stdin closed: leave no force behind
    time.sleep(0.2)


if __name__ == "__main__":
    main()
