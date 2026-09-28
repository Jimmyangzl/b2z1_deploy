"""Read Z1 arm state via unitree_arm_interface."""

import time
from typing import Optional

import numpy as np

from state.arm_defaults import TELEOP_INIT_JOINTS, arm_has_live_signal, arm_state_valid
from utils.math_utils import rotmat_to_quaternion


class Z1StateReader:
    """Z1 arm joint state, FK end-effector pose, and Jacobian.

    EE pose is expressed in the **robot base frame** (origin at B2 base,
    axes aligned with the base). Sport-world odometry is not used.
    """

    def __init__(
        self,
        z1_base_in_robot_base: np.ndarray,
        default_arm_q: Optional[np.ndarray] = None,
    ):
        self.z1_base_in_robot_base = np.asarray(z1_base_in_robot_base, dtype=np.float64)
        self._default_arm_q = (
            TELEOP_INIT_JOINTS.copy()
            if default_arm_q is None
            else np.asarray(default_arm_q, dtype=np.float64).reshape(6).copy()
        )
        self._arm = None
        self._arm_model = None
        self._loop_on = False

    def initialize(self, start_loop: bool = True) -> None:
        """Create ArmInterface. Optionally defer loopOn() until after VR prep releases.

        Starting UDP loop while ``z1_robot_prep`` still holds causes two SDK clients
        to fight; teleop waits for websocket connect (prep release) before loopOn.
        """
        try:
            import unitree_arm_interface

            self._arm = unitree_arm_interface.ArmInterface(hasGripper=True)
            self._arm_model = self._arm._ctrlComp.armModel
            if start_loop:
                self.start_loop()
        except Exception:
            self._arm = None
            self._arm_model = None
            self._loop_on = False

    def start_loop(self) -> None:
        """Start Z1 UDP send/recv. Safe to call once after VR prep has released."""
        if self._arm is None or self._loop_on:
            return
        self._arm.loopOn()
        time.sleep(0.2)
        self._loop_on = True

    def close(self) -> None:
        if self._arm is not None:
            try:
                self._arm.loopOff()
            except Exception:
                pass
            self._loop_on = False

    @property
    def arm(self):
        return self._arm

    @property
    def arm_model(self):
        return self._arm_model

    def _ee_to_base(self, T_fk: np.ndarray):
        """FK in arm-mount frame → robot base (translation-only mount)."""
        ee_pos_arm = np.asarray(T_fk[:3, 3], dtype=np.float64)
        ee_pos_base = self.z1_base_in_robot_base + ee_pos_arm
        ee_quat_base = rotmat_to_quaternion(np.asarray(T_fk[:3, :3], dtype=np.float64))
        return ee_pos_base, ee_quat_base

    @staticmethod
    def _has_arm_signal(q: np.ndarray) -> bool:
        return arm_has_live_signal(q)

    def _fk_from_arm_q(self, arm_q: np.ndarray) -> dict:
        if self._arm_model is None:
            return {
                "ee_pos": np.zeros(3, dtype=np.float64),
                "ee_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                "jacobian": None,
                "T_fk": None,
            }
        T_fk = self._arm_model.forwardKinematics(arm_q, 6)
        ee_pos, ee_quat = self._ee_to_base(T_fk)
        jacobian = np.asarray(self._arm_model.CalcJacobian(arm_q), dtype=np.float64)
        return {
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
            "jacobian": jacobian,
            "T_fk": T_fk,
        }

    def _fallback_read(self) -> dict:
        arm_q = self._default_arm_q.copy()
        arm_dq = np.zeros(6, dtype=np.float64)
        fk = self._fk_from_arm_q(arm_q)
        return {
            "arm_connected": False,
            "arm_q": arm_q,
            "arm_dq": arm_dq,
            "gripper_q": 0.0,
            "ee_pos": fk["ee_pos"],
            "ee_quat": fk["ee_quat"],
            "jacobian": fk["jacobian"],
            "T_fk": fk["T_fk"],
        }

    def read(self) -> dict:
        if self._arm is None or self._arm_model is None:
            return self._fallback_read()

        q = np.asarray(self._arm.lowstate.getQ(), dtype=np.float64).reshape(-1)
        qd = np.asarray(self._arm.lowstate.getQd(), dtype=np.float64).reshape(-1)
        if not self._has_arm_signal(q) or not arm_state_valid(q, qd):
            return self._fallback_read()

        arm_q = q[:6]
        arm_dq = qd[:6] if qd.size >= 6 else np.zeros(6)
        gripper_q = float(q[6]) if q.size > 6 else float(self._arm.lowstate.getGripperQ())
        if not np.isfinite(gripper_q) or abs(gripper_q) > 5.0:
            gripper_q = 0.0

        fk = self._fk_from_arm_q(arm_q)

        return {
            "arm_connected": True,
            "arm_q": arm_q,
            "arm_dq": arm_dq,
            "gripper_q": gripper_q,
            "ee_pos": fk["ee_pos"],
            "ee_quat": fk["ee_quat"],
            "jacobian": fk["jacobian"],
            "T_fk": fk["T_fk"],
        }
