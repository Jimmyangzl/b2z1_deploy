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
    phantom_thresh: float = 0.1,
    phantom_step: float = 0.1,
    rot_phantom_thresh: float = 0.3,
    rot_phantom_step: float = 0.3,
) -> np.ndarray:
    """Build 6D task error ``[drot(3), dpos(3)]`` for Z1 space Jacobian.

    Translation uses the sim/teleop phantom step: if ``||dpos|| > phantom_thresh``,
    scale so the commanded step length is ``phantom_step`` (train thresh=0.1).

    Orientation is likewise capped. Without this, large tracker rotations at
    extreme reaches dominate the DLS solve, starve position recovery, and the
    arm looks permanently stuck near singularities / joint limits.
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
    drot_norm = float(np.linalg.norm(drot))
    if drot_norm > rot_phantom_thresh and drot_norm > 1e-12:
        drot = drot * (rot_phantom_step / drot_norm)

    # SDK space twist order: [omega, v]
    return np.concatenate([drot, dpos]).astype(np.float64)


def incremental_ik_step(
    jacobian: np.ndarray,
    T_ee: np.ndarray,
    T_goal: np.ndarray,
    q: np.ndarray,
    *,
    damping: float = 0.05,
    phantom_thresh: float = 0.1,
    phantom_step: float = 0.1,
    rot_phantom_thresh: float = 0.3,
    rot_phantom_step: float = 0.3,
    dq_clip: float = 0.2,
    elbow_fold_gain: float = 0.25,
    elbow_singularity: float = -0.25,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """One incremental IK step: ``q_goal = q + δq``.

    Damped least squares (teleop_task ``_control_ik``):
        δq = Jᵀ (J Jᵀ + λ² I)⁻¹ dpose,  λ = ``damping``.

    Low manipulability is allowed. Near stretched elbow we only add a null-space
    fold so that once the goal moves back, the arm can leave the singular
    configuration instead of freezing there.

    Returns ``(q_goal, dpose)``. ``q_goal`` is ``None`` if the linear solve fails.
    """
    q = np.asarray(q, dtype=np.float64).reshape(6)
    J = np.asarray(jacobian, dtype=np.float64).reshape(6, 6)
    dpose = pose_error_6d(
        T_ee,
        T_goal,
        phantom_thresh=phantom_thresh,
        phantom_step=phantom_step,
        rot_phantom_thresh=rot_phantom_thresh,
        rot_phantom_step=rot_phantom_step,
    )
    try:
        j_eef_T = J.T
        elbow = float(q[2])
        lam = float(damping)
        # Mild extra damping near stretch — still tracks, but less thrash.
        if elbow > elbow_singularity:
            lam = max(lam, 0.08)
        lam2 = lam ** 2
        A = J @ j_eef_T + np.eye(6, dtype=np.float64) * lam2
        dq = j_eef_T @ np.linalg.solve(A, dpose)

        # Null-space elbow fold: keep teleop alive at low manip, but bias toward
        # a bent elbow so recovery is possible when the goal comes back.
        if elbow > elbow_singularity and elbow_fold_gain > 0.0:
            try:
                A_inv = np.linalg.solve(
                    J @ j_eef_T + np.eye(6) * lam2, np.eye(6)
                )
                N = np.eye(6) - j_eef_T @ A_inv @ J
                q_pref = q.copy()
                q_pref[2] = -0.9
                # Stronger fold when task-space progress would be tiny.
                fold = elbow_fold_gain
                if float(np.linalg.norm(dq)) < 0.02:
                    fold = max(fold, 0.45)
                dq = dq + fold * (N @ (q_pref - q))
            except np.linalg.LinAlgError:
                dq = np.asarray(dq, dtype=np.float64).copy()
                dq[2] -= float(elbow_fold_gain)

        if dq_clip > 0.0:
            dq = np.clip(dq, -dq_clip, dq_clip)
    except np.linalg.LinAlgError:
        return None, dpose
    return (q + dq.reshape(6)).astype(np.float64), dpose
