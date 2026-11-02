"""IMU strapdown base linear velocity with ZUPT.

Integrates gravity-compensated accelerometer readings in a temporary world
frame, then expresses velocity in the body frame. Zero-velocity updates
(ZUPT) fire when the linear accel residual is small **or** leg joint rates
are low (standstill proxies after sport ``ReleaseMode``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from utils.math_utils import quat_rotate_inverse_np, quat_rotate_np


class ImuBaseVelEstimator:
    """Stateful body-frame base velocity from IMU + leg ``dq`` ZUPT."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        c = dict(cfg or {})
        self.gravity = float(c.get("gravity", 9.81))
        self.accel_residual_thresh = float(c.get("accel_residual_thresh", 0.8))
        self.joint_dq_thresh = float(c.get("joint_dq_thresh", 0.25))
        self.max_dt = float(c.get("max_dt", 0.05))
        self.min_dt = float(c.get("min_dt", 1e-4))
        # Soft clamp on integrated speed (m/s) to limit runaway before ZUPT.
        self.max_speed = float(c.get("max_speed", 2.5))

        self._v_world = np.zeros(3, dtype=np.float64)
        self._t_prev: Optional[float] = None
        self._last_zupt = False
        self._last_accel_residual = 0.0
        self._last_joint_dq_norm = 0.0

    def reset(self) -> None:
        self._v_world[:] = 0.0
        self._t_prev = None
        self._last_zupt = False

    @property
    def last_zupt(self) -> bool:
        return bool(self._last_zupt)

    @property
    def last_accel_residual(self) -> float:
        return float(self._last_accel_residual)

    @property
    def last_joint_dq_norm(self) -> float:
        return float(self._last_joint_dq_norm)

    def update(
        self,
        base_quat: np.ndarray,
        accel_body: np.ndarray,
        leg_dq: np.ndarray,
        timestamp: float,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Integrate one IMU sample.

        Parameters
        ----------
        base_quat
            Body→world quaternion (wxyz).
        accel_body
            IMU accelerometer (m/s^2, body frame, includes gravity).
        leg_dq
            Leg joint velocities (sim order, rad/s).
        timestamp
            Monotonic-ish time in seconds (``time.time()`` is fine).

        Returns
        -------
        v_local, v_world, zupt
            Body-frame velocity, world-frame velocity, and whether ZUPT applied.
        """
        q = np.asarray(base_quat, dtype=np.float64).reshape(4)
        accel = np.asarray(accel_body, dtype=np.float64).reshape(3)
        dq = np.asarray(leg_dq, dtype=np.float64).reshape(-1)
        t = float(timestamp)

        # Specific force → linear accel: a = R f + g_world, g_world = [0,0,-g].
        gravity_world = np.array([0.0, 0.0, -self.gravity], dtype=np.float64)
        a_world = quat_rotate_np(q, accel) + gravity_world
        accel_residual = float(np.linalg.norm(a_world))
        joint_dq_norm = float(np.linalg.norm(dq))
        self._last_accel_residual = accel_residual
        self._last_joint_dq_norm = joint_dq_norm

        zupt = (
            accel_residual < self.accel_residual_thresh
            or joint_dq_norm < self.joint_dq_thresh
        )

        if self._t_prev is None:
            self._t_prev = t
            if zupt:
                self._v_world[:] = 0.0
            self._last_zupt = zupt
            v_local = quat_rotate_inverse_np(q, self._v_world)
            return v_local.copy(), self._v_world.copy(), zupt

        dt = t - self._t_prev
        self._t_prev = t
        if dt < self.min_dt or dt > self.max_dt:
            # Skip bad / stalled intervals; keep last velocity (or ZUPT).
            if zupt:
                self._v_world[:] = 0.0
            self._last_zupt = zupt
            v_local = quat_rotate_inverse_np(q, self._v_world)
            return v_local.copy(), self._v_world.copy(), zupt

        if zupt:
            self._v_world[:] = 0.0
        else:
            self._v_world = self._v_world + a_world * dt
            speed = float(np.linalg.norm(self._v_world))
            if speed > self.max_speed > 0.0:
                self._v_world *= self.max_speed / speed

        self._last_zupt = zupt
        v_local = quat_rotate_inverse_np(q, self._v_world)
        return v_local.copy(), self._v_world.copy(), zupt
