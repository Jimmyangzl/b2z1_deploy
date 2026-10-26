"""Convert recorded episodes into arm-base-relative trajectory chunks.

For each start index ``t`` and horizon ``n``, poses in ``[t, t+n)`` are expressed
in the arm-base mount frame at time ``t``. Base motion is recovered by integrating
``base_lin_vel_local * dt`` with ``dt = 1 / record_rate_hz``, rotating each step
into ``F_t`` via ``base_quat``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_ARM_BASE_OFFSET = np.array([0.15, 0.0, 0.10], dtype=np.float64)

# Fields that must be fully finite before transforming.
_NAN_GATE_FIELDS = (
    "ee_goal_local_cart",
    "ee_goal_quat_base",
    "base_lin_vel_local",
    "base_quat",
    "gripper_cmd",
)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    out = np.asarray(q, dtype=np.float64).copy()
    out[..., 1:] *= -1.0
    return out


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product for wxyz quaternions (broadcastable)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) ``v`` by body→world quaternion ``q`` (wxyz)."""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    q_w = q[..., 0]
    q_vec = q[..., 1:]
    a = v * (2.0 * q_w[..., None] ** 2 - 1.0)
    b = np.cross(q_vec, v) * (2.0 * q_w[..., None])
    c = q_vec * np.sum(q_vec * v, axis=-1, keepdims=True) * 2.0
    return a + b + c


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) ``v`` by the inverse of body→world quaternion ``q``."""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    q_w = q[..., 0]
    q_vec = q[..., 1:]
    a = v * (2.0 * q_w[..., None] ** 2 - 1.0)
    b = np.cross(q_vec, v) * (2.0 * q_w[..., None])
    c = q_vec * np.sum(q_vec * v, axis=-1, keepdims=True) * 2.0
    return a - b + c


def report_nan_fields(data: Dict[str, np.ndarray]) -> List[str]:
    """Return human-readable NaN reports for gated fields. Empty ⇒ clean."""
    reports: List[str] = []
    for name in _NAN_GATE_FIELDS:
        arr = np.asarray(data[name])
        bad = ~np.isfinite(arr)
        if not bad.any():
            continue
        if arr.ndim == 1:
            idxs = np.flatnonzero(bad)
        else:
            # row is bad if any component is non-finite
            row_bad = bad.reshape(bad.shape[0], -1).any(axis=1)
            idxs = np.flatnonzero(row_bad)
        preview = idxs[:10].tolist()
        more = "" if idxs.size <= 10 else f" … (+{idxs.size - 10} more)"
        reports.append(
            f"{name}: {idxs.size}/{arr.shape[0]} non-finite frame(s) "
            f"at indices {preview}{more}"
        )
    return reports


def integrate_base_world_positions(
    base_quat: np.ndarray,
    base_lin_vel_local: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Forward-Euler base origins in an arbitrary world frame (p[0]=0)."""
    t_len = int(base_quat.shape[0])
    p = np.zeros((t_len, 3), dtype=np.float64)
    if t_len <= 1:
        return p
    for i in range(t_len - 1):
        p[i + 1] = p[i] + quat_rotate(base_quat[i], base_lin_vel_local[i] * dt)
    return p


def compute_relative_chunks(
    *,
    ee_goal_local_cart: np.ndarray,
    ee_goal_quat_base: np.ndarray,
    base_lin_vel_local: np.ndarray,
    base_quat: np.ndarray,
    gripper_cmd: np.ndarray,
    chunk_n: int,
    dt: float,
    arm_base_offset: Sequence[float] = DEFAULT_ARM_BASE_OFFSET,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``(T, n, …)`` relative EE cart/quat chunks and gripper slices.

    Frame ``F_t`` is the arm-base mount at time ``t`` (same orientation as the
    robot base; origin shifted by ``arm_base_offset`` in the body frame).

    Past the episode end, each horizon step repeats the last relative pose
    (base treated as stationary in the world).
    """
    ee_local = np.asarray(ee_goal_local_cart, dtype=np.float64)
    ee_quat = np.asarray(ee_goal_quat_base, dtype=np.float64)
    vel = np.asarray(base_lin_vel_local, dtype=np.float64)
    quat = np.asarray(base_quat, dtype=np.float64)
    grip = np.asarray(gripper_cmd, dtype=np.float64).reshape(-1)

    t_len = ee_local.shape[0]
    n = int(chunk_n)
    if n < 1:
        raise ValueError(f"chunk_n must be >= 1, got {n}")
    if t_len == 0:
        raise ValueError("episode has zero frames")

    offset = np.asarray(arm_base_offset, dtype=np.float64).reshape(3)
    p_world = integrate_base_world_positions(quat, vel, float(dt))

    cart_chunk = np.empty((t_len, n, 3), dtype=np.float64)
    quat_chunk = np.empty((t_len, n, 4), dtype=np.float64)
    grip_chunk = np.empty((t_len, n), dtype=np.float64)

    for t in range(t_len):
        q_t = quat[t]
        p_arm_t = p_world[t] + quat_rotate(q_t, offset)
        last_cart = np.full(3, np.nan)
        last_quat = np.full(4, np.nan)
        last_grip = np.nan

        for k in range(n):
            i = t + k
            if i >= t_len:
                cart_chunk[t, k] = last_cart
                quat_chunk[t, k] = last_quat
                grip_chunk[t, k] = last_grip
                continue

            ee_world = p_world[i] + quat_rotate(quat[i], ee_local[i] + offset)
            ee_rel = quat_rotate_inverse(q_t, ee_world - p_arm_t)
            q_ee_rel = quat_mul(quat_conjugate(q_t), quat_mul(quat[i], ee_quat[i]))
            # Keep unit length for numerical drift from mul chain.
            q_norm = np.linalg.norm(q_ee_rel)
            if q_norm > 0.0:
                q_ee_rel = q_ee_rel / q_norm

            g = float(grip[i])
            cart_chunk[t, k] = ee_rel
            quat_chunk[t, k] = q_ee_rel
            grip_chunk[t, k] = g
            last_cart = ee_rel
            last_quat = q_ee_rel
            last_grip = g

    return cart_chunk, quat_chunk, grip_chunk


def transform_episode_arrays(
    data: Dict[str, np.ndarray],
    *,
    chunk_n: int,
    record_rate_hz: float,
    arm_base_offset: Sequence[float] = DEFAULT_ARM_BASE_OFFSET,
) -> Optional[Dict[str, np.ndarray]]:
    """Return relative-traj arrays, or ``None`` if NaN gate fails.

    On failure, prints a report for the caller to attribute to an episode path.
    """
    reports = report_nan_fields(data)
    if reports:
        for line in reports:
            print(f"  NaN gate: {line}", flush=True)
        return None

    rate = float(record_rate_hz)
    if rate <= 0.0:
        raise ValueError(f"record_rate_hz must be > 0, got {rate}")
    dt = 1.0 / rate

    cart, quat, grip = compute_relative_chunks(
        ee_goal_local_cart=data["ee_goal_local_cart"],
        ee_goal_quat_base=data["ee_goal_quat_base"],
        base_lin_vel_local=data["base_lin_vel_local"],
        base_quat=data["base_quat"],
        gripper_cmd=data["gripper_cmd"],
        chunk_n=chunk_n,
        dt=dt,
        arm_base_offset=arm_base_offset,
    )
    return {
        "timestamp": np.asarray(data["timestamp"], dtype=np.float64),
        "rgb": np.asarray(data["rgb"]),
        "ee_goal_cart_chunk": cart,
        "ee_goal_quat_chunk": quat,
        "gripper_cmd": grip,
    }
