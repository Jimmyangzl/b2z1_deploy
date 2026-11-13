#!/usr/bin/env python3
"""Verify non-blocking obs/VR SHM locks (keep-last on contention).

Locks are multiprocessing RLocks, so contention must be created from another
process (same-process re-acquire always succeeds).
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from paths import setup_paths

setup_paths()

from obs.process_obs_worker import ProcessObsWorker
from state.aggregator import load_config
from state.robot_state import RobotState
from vr_goal.process_vr_client import ProcessVrWorker


def _hold_array_lock(arr, ready_evt, release_evt) -> None:
    lock = arr.get_lock()
    lock.acquire()
    ready_evt.set()
    release_evt.wait(10.0)
    lock.release()


def main() -> None:
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    cfg = load_config(os.path.join(ROOT, "config/b2z1_wbc.yaml"))
    w = ProcessObsWorker(cfg, control_dt=0.02, use_vr=False)
    w.start()
    w.wait_ready(60.0)
    robot = RobotState(valid=True, arm_connected=True)
    robot.jacobian = np.eye(6)

    assert w.submit(robot, None) is True
    time.sleep(0.05)
    pack = None
    for _ in range(50):
        pack = w.take()
        if pack is not None:
            break
        time.sleep(0.01)
    assert pack is not None, "warmup take failed"

    ready = ctx.Event()
    release = ctx.Event()
    holder = ctx.Process(
        target=_hold_array_lock, args=(w._in, ready, release), daemon=True
    )
    holder.start()
    assert ready.wait(5.0), "in-lock holder did not start"
    t0 = time.perf_counter()
    ok = w.submit(robot, None)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    release.set()
    holder.join(timeout=2.0)
    assert ok is False, "submit should fail while lock held by other process"
    assert dt_ms < 5.0, f"submit blocked too long: {dt_ms:.1f}ms"
    print(f"obs submit try-lock OK ({dt_ms:.2f}ms)")

    assert w.submit(robot, None) is True
    deadline = time.time() + 2.0
    done = w._last_taken
    while time.time() < deadline:
        with w._done_seq.get_lock():
            done = int(w._done_seq.value)
        if done > w._last_taken:
            break
        time.sleep(0.01)
    assert done > w._last_taken, "child did not finish"

    ready = ctx.Event()
    release = ctx.Event()
    holder = ctx.Process(
        target=_hold_array_lock, args=(w._out, ready, release), daemon=True
    )
    holder.start()
    assert ready.wait(5.0), "out-lock holder did not start"
    t0 = time.perf_counter()
    got = w.take()
    dt_ms = (time.perf_counter() - t0) * 1000.0
    release.set()
    holder.join(timeout=2.0)
    assert got is None, "take should return None while out lock held"
    assert dt_ms < 5.0, f"take blocked too long: {dt_ms:.1f}ms"
    print(f"obs take try-lock OK ({dt_ms:.2f}ms)")

    w.push_action(np.arange(12, dtype=np.float64))
    ready = ctx.Event()
    release = ctx.Event()
    holder = ctx.Process(
        target=_hold_array_lock, args=(w._in, ready, release), daemon=True
    )
    holder.start()
    assert ready.wait(5.0), "in-lock holder did not start"
    assert w.submit(robot, None) is False
    release.set()
    holder.join(timeout=2.0)
    assert w._pending_action is not None, "pending action dropped on failed submit"
    print("pending action retained OK")

    w.stop()

    vr = ProcessVrWorker(host="127.0.0.1", port=8765)
    vr._pose_cache = {
        "connected": True,
        "holding": False,
        "robot_ready": False,
        "timestamp": 1.0,
        "position": np.zeros(3),
        "orientation": None,
        "Tdes": np.eye(4),
        "T_init": None,
        "q_init": None,
    }
    vr._last_seq = 0
    with vr._seq.get_lock():
        vr._seq.value = 1
    ready = ctx.Event()
    release = ctx.Event()
    holder = ctx.Process(
        target=_hold_array_lock, args=(vr._pose, ready, release), daemon=True
    )
    holder.start()
    assert ready.wait(5.0), "pose-lock holder did not start"
    t0 = time.perf_counter()
    pose = vr.get_latest_pose()
    dt_ms = (time.perf_counter() - t0) * 1000.0
    release.set()
    holder.join(timeout=2.0)
    assert pose is vr._pose_cache
    assert dt_ms < 5.0, f"VR get_latest_pose blocked: {dt_ms:.1f}ms"
    print(f"VR pose try-lock OK ({dt_ms:.2f}ms)")

    vr2 = ProcessVrWorker(host="127.0.0.1", port=8766)
    ready = ctx.Event()
    release = ctx.Event()
    holder = ctx.Process(
        target=_hold_array_lock, args=(vr2._pose, ready, release), daemon=True
    )
    holder.start()
    assert ready.wait(5.0), "pose-lock holder did not start"
    t0 = time.perf_counter()
    pose = vr2.get_latest_pose()
    dt_ms = (time.perf_counter() - t0) * 1000.0
    release.set()
    holder.join(timeout=2.0)
    assert pose["connected"] is False and pose["Tdes"] is None
    assert dt_ms < 5.0
    print(f"VR empty fallback OK ({dt_ms:.2f}ms)")
    print("ALL VERIFY OK")


if __name__ == "__main__":
    main()
