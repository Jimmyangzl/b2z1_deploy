"""Default Z1 arm pose when live arm state is unavailable."""

import math
from typing import Optional

import numpy as np

# Fallback when config arm_init_joint is unavailable.
# Keep in sync with b2/wbc_deploy/config/b2z1_wbc.yaml::arm_init_joint.
TELEOP_INIT_JOINTS = np.array(
    [0.0, 0.6, -0.6, 0.0, 0.0, math.pi / 2.0],
    dtype=np.float64,
)

# Loose bounds for rejecting corrupt SDK feedback (real joints are within ~±3 rad).
MAX_ARM_Q_ABS = 10.0
MAX_ARM_DQ_ABS = 50.0
MAX_GRIPPER_Q_ABS = 5.0


def arm_q_valid(q: np.ndarray) -> bool:
    q6 = np.asarray(q, dtype=np.float64).reshape(-1)[:6]
    if q6.size < 6 or not np.all(np.isfinite(q6)):
        return False
    return bool(np.max(np.abs(q6)) <= MAX_ARM_Q_ABS)


def arm_dq_valid(qd: np.ndarray) -> bool:
    qd6 = np.asarray(qd, dtype=np.float64).reshape(-1)[:6]
    if qd6.size < 6 or not np.all(np.isfinite(qd6)):
        return False
    return bool(np.max(np.abs(qd6)) <= MAX_ARM_DQ_ABS)


def arm_state_valid(q: np.ndarray, qd: Optional[np.ndarray] = None) -> bool:
    if not arm_q_valid(q):
        return False
    if qd is None:
        return True
    return arm_dq_valid(qd)


def arm_has_live_signal(q: np.ndarray) -> bool:
    """Non-trivial, in-range joint feedback (z1_ctrl running with sane state)."""
    q6 = np.asarray(q, dtype=np.float64).reshape(-1)[:6]
    return arm_q_valid(q) and bool(np.max(np.abs(q6)) > 1e-4)
