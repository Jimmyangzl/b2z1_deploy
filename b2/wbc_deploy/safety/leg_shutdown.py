"""Soft-land B2 legs before releasing lowcmd (damping)."""

from __future__ import annotations

import signal
import time
from typing import Optional

import numpy as np


def leg_joint_dict_to_array(cfg: dict, key: str) -> np.ndarray:
    """Build 12-dim sim-order legs from config ``{FL,FR,RL,RR: [hip,thigh,calf]}``."""
    block = cfg.get(key)
    if block is None:
        raise KeyError(f"Missing {key} in config (expected FL/FR/RL/RR × [hip, thigh, calf])")
    order = ("FL", "FR", "RL", "RR")
    out = []
    for leg in order:
        if leg not in block:
            raise KeyError(f"{key} missing leg {leg}")
        vals = np.asarray(block[leg], dtype=np.float64).reshape(-1)
        if vals.size != 3:
            raise ValueError(f"{key}.{leg} must have 3 values (hip, thigh, calf), got {vals.size}")
        out.extend(vals.tolist())
    return np.asarray(out, dtype=np.float64)


def run_leg_soft_shutdown(
    *,
    lowcmd_writer,
    aggregator,
    q_goal: np.ndarray,
    ramp_s: float = 3.0,
    hold_s: float = 0.5,
    rate_hz: float = 50.0,
    kp: float,
    kd: float,
    q_start: Optional[np.ndarray] = None,
) -> None:
    """Interpolate legs to ``q_goal``, hold briefly, then return (caller stops lowcmd).

    Uses policy-scale PD gains. A second Ctrl+C aborts the ramp and returns early
    so the caller can stop lowcmd immediately (damping).
    """
    q_goal = np.asarray(q_goal, dtype=np.float64).reshape(12)
    if q_start is None:
        robot = aggregator.read()
        if robot.valid:
            q_start = np.asarray(robot.leg_q[:12], dtype=np.float64).copy()
        else:
            q_start = np.asarray(lowcmd_writer.get_leg_targets_sim(), dtype=np.float64).copy()
    else:
        q_start = np.asarray(q_start, dtype=np.float64).reshape(12)

    ramp_s = max(0.0, float(ramp_s))
    hold_s = max(0.0, float(hold_s))
    rate_hz = max(1.0, float(rate_hz))
    dt = 1.0 / rate_hz

    lowcmd_writer.set_gains(float(kp), float(kd))

    abort = {"flag": False}
    prev_int = signal.getsignal(signal.SIGINT)
    prev_term = signal.getsignal(signal.SIGTERM)

    def _on_signal(signum, _frame) -> None:
        if abort["flag"]:
            print(
                f"\nSignal {signum}: aborting soft shutdown → damping now.",
                flush=True,
            )
            raise KeyboardInterrupt
        abort["flag"] = True
        print(
            f"\nSignal {signum}: soft shutdown in progress "
            "(Ctrl+C again to abort to damping).",
            flush=True,
        )

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        print(
            f"\n[safety] Soft shutdown: ramp {ramp_s:.1f}s → hold {hold_s:.1f}s "
            f"at lay-down (Kp={kp}, Kd={kd})",
            flush=True,
        )
        print(
            f"[safety]   q_start: {np.array2string(q_start, precision=3)}",
            flush=True,
        )
        print(
            f"[safety]   q_goal : {np.array2string(q_goal, precision=3)}",
            flush=True,
        )

        ramp_steps = max(1, int(round(ramp_s * rate_hz))) if ramp_s > 0 else 0
        for i in range(ramp_steps):
            alpha = (i + 1) / ramp_steps
            q_cmd = (1.0 - alpha) * q_start + alpha * q_goal
            lowcmd_writer.set_leg_targets(q_cmd)
            time.sleep(dt)

        lowcmd_writer.set_leg_targets(q_goal)
        if hold_s > 0:
            hold_end = time.perf_counter() + hold_s
            while time.perf_counter() < hold_end:
                lowcmd_writer.set_leg_targets(q_goal)
                time.sleep(min(dt, hold_end - time.perf_counter()))

        print("[safety] Soft shutdown complete; stopping lowcmd (damping).", flush=True)
    except KeyboardInterrupt:
        print("[safety] Soft shutdown aborted.", flush=True)
    finally:
        signal.signal(signal.SIGINT, prev_int)
        signal.signal(signal.SIGTERM, prev_term)
