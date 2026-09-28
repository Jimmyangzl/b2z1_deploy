"""VR WebSocket + VrPoseMapper wrapper for WBC EE goals (robot base frame)."""

from typing import Optional

import numpy as np

from teleop.vr_pose_mapper import VrPoseMapper
from teleop.vr_websocket_client import VrWebSocketClient

from utils.math_utils import (
    quat_apply,
    quat_conjugate,
    quat_mul,
    quat_rotate_inverse_np,
    quaternion_to_rpy,
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

    Internally, tracker deltas are applied in a gravity-aligned frame whose
    origin coincides with the robot base (no sport odom), then converted back
    to the robot base frame for obs / IK.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8765, config_path: Optional[str] = None):
        self.client = VrWebSocketClient(host=host, port=port, config_path=config_path)
        self.mapper = VrPoseMapper(config_path=config_path)
        self.curr_ee_goal_cart_base = None
        self.ee_goal_orn_quat = None
        self.ee_pos_goal_base = None
        self.ee_quat_goal_base = None
        self._started = False

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

    @property
    def is_active(self) -> bool:
        return self.mapper.mode == self.mapper.MODE_ACTIVE

    @property
    def ee_pos_goal_world(self):
        return self.ee_pos_goal_base

    @property
    def ee_quat_goal_world(self):
        return self.ee_quat_goal_base

    def update(self, ee_pos, ee_quat, base_quat) -> None:
        """``ee_pos`` / ``ee_quat`` are in the robot base frame."""
        ee_pos = np.asarray(ee_pos, dtype=np.float64)
        ee_quat = np.asarray(ee_quat, dtype=np.float64)
        base_quat = np.asarray(base_quat, dtype=np.float64)
        base_yaw = quaternion_to_rpy(base_quat)[2]

        # Mapper expects gravity-aligned positions with origin at the base
        # (equivalent to old world with base_pos ≡ 0).
        ee_pos_aligned = quat_apply(base_quat, ee_pos)
        ee_quat_aligned = quat_mul(base_quat, ee_quat)

        vr_pose = self.client.get_latest_pose()
        goal_aligned, goal_quat_aligned = self.mapper.update(
            vr_pose, ee_pos_aligned, ee_quat_aligned, base_yaw
        )

        goal_base = quat_rotate_inverse_np(base_quat, goal_aligned)
        goal_quat_base = quat_mul(quat_conjugate(base_quat), goal_quat_aligned)

        self.curr_ee_goal_cart_base = goal_base.copy()
        self.ee_goal_orn_quat = goal_quat_base.copy()
        self.ee_pos_goal_base = goal_base.copy()
        self.ee_quat_goal_base = goal_quat_base.copy()

    def get_goal_position(self, fallback_ee_pos):
        if self.curr_ee_goal_cart_base is None:
            return fallback_ee_pos.copy()
        return self.curr_ee_goal_cart_base.copy()
