"""VR WebSocket wrapper for WBC EE goals (robot base frame).

Deploy arm control uses the Tdes teleop law
``T_goal = T_init @ inv(Tdes0) @ Tdes`` (arm-base / FK frame). Policy obs
must use that same target — not VrPoseMapper deltas — otherwise
``ee_goal_local_cart`` understates reach while measured EE / ``dist_base_ee``
look far (Tdes includes ``translational_gain`` and full SE3).
"""

from typing import Optional

import numpy as np

from teleop.vr_pose_mapper import VrPoseMapper
from teleop.vr_websocket_client import VrWebSocketClient

from utils.math_utils import (
    pos_quat_to_homogeneous,
    quat_conjugate,
    quat_mul,
    quat_apply,
    quat_rotate_inverse_np,
    quaternion_to_rpy,
    rotmat_to_quaternion,
)


class NullVrGoalProvider:
    """VR disabled: hold EE goal at the current end-effector pose (base frame)."""

    MODE_IDLE = "idle"

    def __init__(self):
        self.curr_ee_goal_cart_base = None
        self.ee_goal_orn_quat = None
        self.ee_pos_goal_base = None
        self.ee_quat_goal_base = None
        self.mode = self.MODE_IDLE

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def reset(self) -> None:
        self.curr_ee_goal_cart_base = None
        self.ee_goal_orn_quat = None
        self.ee_pos_goal_base = None
        self.ee_quat_goal_base = None

    @property
    def is_active(self) -> bool:
        return False

    # Back-compat aliases used by older call sites / prints.
    @property
    def ee_pos_goal_world(self):
        return self.ee_pos_goal_base

    @property
    def ee_quat_goal_world(self):
        return self.ee_quat_goal_base

    def update(self, ee_pos, ee_quat, base_quat) -> None:
        del base_quat
        self.curr_ee_goal_cart_base = np.asarray(ee_pos, dtype=np.float64).copy()
        self.ee_goal_orn_quat = np.asarray(ee_quat, dtype=np.float64).copy()
        self.ee_pos_goal_base = self.curr_ee_goal_cart_base.copy()
        self.ee_quat_goal_base = self.ee_goal_orn_quat.copy()

    def get_goal_position(self, fallback_ee_pos):
        if self.curr_ee_goal_cart_base is None:
            return np.asarray(fallback_ee_pos, dtype=np.float64).copy()
        return self.curr_ee_goal_cart_base.copy()


class VrGoalProvider:
    """Map Vive tracker stream to robot-base-frame end-effector goals.

    Primary path (matches Z1ArmWriter ``tdes`` mode):
      ``T_goal_arm = T_init @ inv(Tdes0) @ Tdes``
      ``goal_base = T_goal_arm[:3,3] + arm_mount``

    Fallback when Tdes is unavailable: VrPoseMapper deltas in a gravity-aligned
    frame, then rotated back into the robot base frame.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        config_path: Optional[str] = None,
        arm_mount_in_base: Optional[np.ndarray] = None,
    ):
        self.client = VrWebSocketClient(host=host, port=port, config_path=config_path)
        self.mapper = VrPoseMapper(config_path=config_path)
        self.arm_mount_in_base = np.asarray(
            [0.03, 0.0, 0.09] if arm_mount_in_base is None else arm_mount_in_base,
            dtype=np.float64,
        ).reshape(3)
        self.curr_ee_goal_cart_base = None
        self.ee_goal_orn_quat = None
        self.ee_pos_goal_base = None
        self.ee_quat_goal_base = None
        self._started = False
        self._T_init: Optional[np.ndarray] = None
        self._Tdes0: Optional[np.ndarray] = None
        self._goal_source = "idle"

    def start(self) -> None:
        if not self._started:
            self.client.start()
            self._started = True

    def stop(self) -> None:
        if self._started:
            self.client.stop()
            self._started = False

    def reset(self) -> None:
        self.mapper.reset()
        self.curr_ee_goal_cart_base = None
        self.ee_goal_orn_quat = None
        self.ee_pos_goal_base = None
        self.ee_quat_goal_base = None
        self._T_init = None
        self._Tdes0 = None
        self._goal_source = "idle"

    @property
    def is_active(self) -> bool:
        return self._goal_source == "tdes" or self.mapper.mode == self.mapper.MODE_ACTIVE

    @property
    def ee_pos_goal_world(self):
        return self.ee_pos_goal_base

    @property
    def ee_quat_goal_world(self):
        return self.ee_quat_goal_base

    def _set_goal_base(self, goal_base: np.ndarray, goal_quat_base: np.ndarray) -> None:
        self.curr_ee_goal_cart_base = np.asarray(goal_base, dtype=np.float64).reshape(3).copy()
        self.ee_goal_orn_quat = np.asarray(goal_quat_base, dtype=np.float64).reshape(4).copy()
        self.ee_pos_goal_base = self.curr_ee_goal_cart_base.copy()
        self.ee_quat_goal_base = self.ee_goal_orn_quat.copy()

    def _hold_at_ee(self, ee_pos: np.ndarray, ee_quat: np.ndarray) -> None:
        # Keep T_init/Tdes0 across brief VR loss (same as Z1ArmWriter). Clearing
        # them after an extreme reach re-locks to "stay at extreme" when the
        # tracker is already drawn back.
        self._goal_source = "hold"
        self._set_goal_base(ee_pos, ee_quat)

    def _update_from_tdes(
        self,
        vr_pose: dict,
        ee_pos: np.ndarray,
        ee_quat: np.ndarray,
    ) -> bool:
        """Return True if goal was set from Tdes teleop law."""
        connected = bool(vr_pose.get("connected", False))
        Tdes = vr_pose.get("Tdes")
        if not connected or Tdes is None:
            return False

        Tdes = np.asarray(Tdes, dtype=np.float64).reshape(4, 4)
        ee_arm = np.asarray(ee_pos, dtype=np.float64).reshape(3) - self.arm_mount_in_base
        T_ee = pos_quat_to_homogeneous(ee_arm, ee_quat)

        # Same engage rule as Z1ArmWriter: lock both baselines at "now".
        if self._T_init is None or self._Tdes0 is None:
            self._T_init = T_ee.copy()
            self._Tdes0 = Tdes.copy()

        T_rel = np.linalg.inv(self._Tdes0) @ Tdes
        T_goal = self._T_init @ T_rel
        pos_arm = np.asarray(T_goal[:3, 3], dtype=np.float64).reshape(3)
        quat = rotmat_to_quaternion(np.asarray(T_goal[:3, :3], dtype=np.float64))
        n = float(np.linalg.norm(quat))
        if n > 1e-12:
            quat = quat / n

        self._goal_source = "tdes"
        self._set_goal_base(pos_arm + self.arm_mount_in_base, quat)
        return True

    def _update_from_mapper(
        self,
        vr_pose: dict,
        ee_pos: np.ndarray,
        ee_quat: np.ndarray,
        base_quat: np.ndarray,
    ) -> None:
        """Legacy mapper path (gravity-aligned deltas → robot base)."""
        base_yaw = quaternion_to_rpy(base_quat)[2]
        ee_pos_aligned = quat_apply(base_quat, ee_pos)
        ee_quat_aligned = quat_mul(base_quat, ee_quat)

        goal_aligned, goal_quat_aligned = self.mapper.update(
            vr_pose, ee_pos_aligned, ee_quat_aligned, base_yaw
        )
        goal_base = quat_rotate_inverse_np(base_quat, goal_aligned)
        goal_quat_base = quat_mul(quat_conjugate(base_quat), goal_quat_aligned)

        self._goal_source = "mapper"
        self._set_goal_base(goal_base, goal_quat_base)

    def update(self, ee_pos, ee_quat, base_quat) -> None:
        """``ee_pos`` / ``ee_quat`` are in the robot base frame."""
        ee_pos = np.asarray(ee_pos, dtype=np.float64)
        ee_quat = np.asarray(ee_quat, dtype=np.float64)
        base_quat = np.asarray(base_quat, dtype=np.float64)

        vr_pose = self.client.get_latest_pose()
        if self._update_from_tdes(vr_pose, ee_pos, ee_quat):
            return

        # No Tdes: hold at EE if tracker down; otherwise mapper fallback.
        if not bool(vr_pose.get("connected", False)):
            self.mapper.reset()
            self._hold_at_ee(ee_pos, ee_quat)
            return

        self._update_from_mapper(vr_pose, ee_pos, ee_quat, base_quat)

    def get_goal_position(self, fallback_ee_pos):
        if self.curr_ee_goal_cart_base is None:
            return np.asarray(fallback_ee_pos, dtype=np.float64).copy()
        return self.curr_ee_goal_cart_base.copy()
