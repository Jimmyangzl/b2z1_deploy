"""EE goal provider driven by ACT WebSocket goals (arm-base → robot-base)."""

from __future__ import annotations

from typing import Optional

import numpy as np


class ActGoalProvider:
    """Sample latest ACT goal for Z1 IK / WBC obs (hold-last between ACT ticks).

    ACT publishes ``ee_goal_local_cart`` / ``ee_goal_quat_base`` in the **arm-base**
    frame. ``get_goal_position`` returns robot-base (arm-base + mount offset),
    matching ``VrGoalProvider``.
    """

    MODE_IDLE = "idle"
    MODE_ACT = "act"
    MODE_HOLD = "hold"

    def __init__(
        self,
        act_server,
        arm_mount_in_base: Optional[np.ndarray] = None,
    ):
        self.server = act_server
        self.arm_mount_in_base = np.asarray(
            [0.03, 0.0, 0.09] if arm_mount_in_base is None else arm_mount_in_base,
            dtype=np.float64,
        ).reshape(3)
        self.curr_ee_goal_cart_base = None
        self.ee_goal_orn_quat = None
        self.ee_pos_goal_base = None
        self.ee_quat_goal_base = None
        self.gripper_cmd = 0.0
        self.mode = self.MODE_IDLE
        self._last_seq = -1
        # Deploy enables only after policy YES (init hold confirm).
        self._enabled = False

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def set_enabled(self, enabled: bool) -> None:
        """When False, ignore ACT stream (hold arm / obs at measured EE)."""
        was = self._enabled
        self._enabled = bool(enabled)
        if self._enabled and not was:
            print(
                "[act-ee] ACT EE goals ENABLED (policy confirmed).",
                flush=True,
            )
        elif (not self._enabled) and was:
            print("[act-ee] ACT EE goals disabled.", flush=True)

    def reset(self) -> None:
        self.curr_ee_goal_cart_base = None
        self.ee_goal_orn_quat = None
        self.ee_pos_goal_base = None
        self.ee_quat_goal_base = None
        self.gripper_cmd = 0.0
        self.mode = self.MODE_IDLE
        self._last_seq = -1

    @property
    def is_active(self) -> bool:
        return self._enabled and self.mode == self.MODE_ACT

    @property
    def client(self):
        """No VR client; Z1ArmWriter VR paths should not use this."""
        return None

    @property
    def ee_pos_goal_world(self):
        return self.ee_pos_goal_base

    @property
    def ee_quat_goal_world(self):
        return self.ee_quat_goal_base

    def latest_arm_goal(self) -> Optional[dict]:
        """Arm-base goal for Z1 IK, or None if disabled / no valid ACT goal yet."""
        if not self._enabled:
            return None
        snap = self.server.get_latest()
        if not snap.get("valid", False):
            return None
        return {
            "ee_goal_local_cart": np.asarray(
                snap["ee_goal_local_cart"], dtype=np.float64
            ).reshape(3),
            "ee_goal_quat_base": np.asarray(
                snap["ee_goal_quat_base"], dtype=np.float64
            ).reshape(4),
            "gripper_cmd": float(snap.get("gripper_cmd", 0.0)),
            "seq": int(snap.get("seq", 0)),
        }

    def update(self, ee_pos, ee_quat, base_quat) -> None:
        del base_quat
        arm = self.latest_arm_goal()
        if arm is None:
            # Hold last goal if we ever had one; else track measured EE.
            if self.curr_ee_goal_cart_base is None:
                self.mode = self.MODE_HOLD
                ee_pos = np.asarray(ee_pos, dtype=np.float64).reshape(3)
                ee_quat = np.asarray(ee_quat, dtype=np.float64).reshape(4)
                self.curr_ee_goal_cart_base = ee_pos.copy()
                self.ee_goal_orn_quat = ee_quat.copy()
                self.ee_pos_goal_base = ee_pos.copy()
                self.ee_quat_goal_base = ee_quat.copy()
            else:
                self.mode = self.MODE_HOLD
            return

        cart_arm = arm["ee_goal_local_cart"]
        quat = arm["ee_goal_quat_base"]
        goal_base = cart_arm + self.arm_mount_in_base
        self.curr_ee_goal_cart_base = goal_base.copy()
        self.ee_goal_orn_quat = quat.copy()
        self.ee_pos_goal_base = goal_base.copy()
        self.ee_quat_goal_base = quat.copy()
        if np.isfinite(arm["gripper_cmd"]):
            self.gripper_cmd = float(arm["gripper_cmd"])
        self.mode = self.MODE_ACT
        self._last_seq = int(arm["seq"])

    def get_goal_position(self, fallback_ee_pos):
        if self.curr_ee_goal_cart_base is None:
            return np.asarray(fallback_ee_pos, dtype=np.float64).copy()
        return self.curr_ee_goal_cart_base.copy()
