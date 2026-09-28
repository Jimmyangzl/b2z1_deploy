"""Incremental (damped least-squares) IK matching teleop_task / sim.

Z1 SDK ``CalcJacobian`` is a **space** Jacobian with twist order ``[ω; v]``
(see unitreeArm.h: twist = (omega, v); example_model.py). MuJoCo ``mj_jacSite``
uses ``[v; ω]``. This module builds ``dpose = [drot; dpos]`` for CalcJacobian.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from utils.math_utils import orientation_error, rotmat_to_quaternion


def pose_error_6d(
    T_ee: np.ndarray,
    T_goal: np.ndarray,
    *,
    phantom_thresh: float = 0.2,
    phantom_step: float = 0.1,
) -> np.ndarray:
    """Build 6D task error ``[drot(3), dpos(3)]`` for Z1 space Jacobian.

    Translation uses the teleop_task phantom step: if ``||dpos|| > phantom_thresh``,
    scale so the commanded step length is ``phantom_step``.
    """
    T_ee = np.asarray(T_ee, dtype=np.float64).reshape(4, 4)
    T_goal = np.asarray(T_goal, dtype=np.float64).reshape(4, 4)

    dpos = T_goal[:3, 3] - T_ee[:3, 3]
    dpos_norm = float(np.linalg.norm(dpos))
    if dpos_norm > phantom_thresh and dpos_norm > 1e-12:
        dpos = dpos * (phantom_step / dpos_norm)

    quat_goal = rotmat_to_quaternion(T_goal[:3, :3])
    quat_ee = rotmat_to_quaternion(T_ee[:3, :3])
    quat_ee = quat_ee / max(np.linalg.norm(quat_ee), 1e-12)
    drot = orientation_error(quat_goal, quat_ee)
    # SDK space twist order: [omega, v]
    return np.concatenate([drot, dpos]).astype(np.float64)


def incremental_ik_step(
    jacobian: np.ndarray,
    T_ee: np.ndarray,
    T_goal: np.ndarray,
    q: np.ndarray,
    *,
    damping: float = 0.05,
    phantom_thresh: float = 0.2,
    phantom_step: float = 0.1,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """One incremental IK step: ``q_goal = q + δq``.

    Damped least squares (teleop_task ``_control_ik``):
        δq = Jᵀ (J Jᵀ + λ² I)⁻¹ dpose,  λ = ``damping``.

    Returns ``(q_goal, dpose)``. ``q_goal`` is ``None`` if the linear solve fails.
    """
    q = np.asarray(q, dtype=np.float64).reshape(6)
    J = np.asarray(jacobian, dtype=np.float64).reshape(6, 6)
    dpose = pose_error_6d(
        T_ee,
        T_goal,
        phantom_thresh=phantom_thresh,
        phantom_step=phantom_step,
    )
    try:
        j_eef_T = J.T
        lam2 = float(damping) ** 2
        A = J @ j_eef_T + np.eye(6, dtype=np.float64) * lam2
        dq = j_eef_T @ np.linalg.solve(A, dpose)
    except np.linalg.LinAlgError:
        return None, dpose
    return (q + dq.reshape(6)).astype(np.float64), dpose
