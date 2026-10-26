"""Hard joint-position clamps matching training ``hard_dof_pos_limit``."""

from __future__ import annotations

from typing import Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from state.leg_dof_mapping import SIM_LEG_DOF_NAMES

# Matches b2z1_lambdawbc_config.control.hard_dof_pos_limit (gripper omitted).
DEFAULT_HARD_DOF_POS_LIMIT: Dict[str, Tuple[float, float]] = {
    "FL_hip_joint": (-0.2, 0.2),
    "FL_thigh_joint": (-0.94, 1.0),
    "FL_calf_joint": (-2.82, -1.0),
    "FR_hip_joint": (-0.2, 0.2),
    "FR_thigh_joint": (-0.94, 1.0),
    "FR_calf_joint": (-2.82, -1.0),
    "RL_hip_joint": (-0.2, 0.2),
    "RL_thigh_joint": (-0.94, 1.0),
    "RL_calf_joint": (-2.82, -1.0),
    "RR_hip_joint": (-0.2, 0.2),
    "RR_thigh_joint": (-0.94, 1.0),
    "RR_calf_joint": (-2.82, -1.0),
    "z1_waist": (-2.5, 2.5),
    "z1_shoulder": (0.0, 2.8),
    "z1_elbow": (-2.5, 0.0),
    # "z1_elbow": (-2.7, 0.0),
    "z1_wrist_angle": (-1.5, 1.5),
    "z1_forearm_roll": (-1.5, 1.5),
    "z1_wrist_rotate": (-2.5, 2.5),
}

ARM_DOF_NAMES = (
    "z1_waist",
    "z1_shoulder",
    "z1_elbow",
    "z1_wrist_angle",
    "z1_forearm_roll",
    "z1_wrist_rotate",
)


def _as_lo_hi(value: Sequence[float], name: str) -> Tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} limit must be [lower, upper], got {value!r}")
    lo, hi = float(value[0]), float(value[1])
    if lo > hi:
        raise ValueError(f"{name} limit lower > upper: ({lo}, {hi})")
    return lo, hi


def merge_hard_dof_pos_limit(
    override: Optional[Mapping[str, Sequence[float]]] = None,
) -> Dict[str, Tuple[float, float]]:
    """Return full hard limits; ``override`` replaces named joints."""
    out: Dict[str, Tuple[float, float]] = dict(DEFAULT_HARD_DOF_POS_LIMIT)
    if not override:
        return out
    for name, lim in override.items():
        out[str(name)] = _as_lo_hi(lim, str(name))
    return out


def limits_arrays(
    hard: Mapping[str, Tuple[float, float]],
    names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    lo = np.empty(len(names), dtype=np.float64)
    hi = np.empty(len(names), dtype=np.float64)
    for i, name in enumerate(names):
        if name not in hard:
            raise KeyError(f"missing hard_dof_pos_limit for {name}")
        lo[i], hi[i] = hard[name]
    return lo, hi


def clip_dof_pos(q: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.shape[0] != lo.shape[0]:
        raise ValueError(f"q length {q.shape[0]} != limits length {lo.shape[0]}")
    return np.clip(q, lo, hi)


def load_leg_arm_limit_arrays(
    config: Optional[MutableMapping] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(leg_lo, leg_hi, arm_lo, arm_hi)`` from config or defaults."""
    cfg = config or {}
    control = cfg.get("control") or {}
    override = control.get("hard_dof_pos_limit")
    # Back-compat: old elbow-only yaml key.
    arm_legacy = cfg.get("arm_joint_limits") or {}
    if override is None and "elbow" in arm_legacy:
        override = {"z1_elbow": arm_legacy["elbow"]}
    hard = merge_hard_dof_pos_limit(override)
    leg_lo, leg_hi = limits_arrays(hard, SIM_LEG_DOF_NAMES)
    arm_lo, arm_hi = limits_arrays(hard, ARM_DOF_NAMES)
    return leg_lo, leg_hi, arm_lo, arm_hi
