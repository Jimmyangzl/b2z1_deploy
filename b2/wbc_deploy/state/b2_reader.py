"""Read B2 low-level and sport mode state over DDS."""

import threading
import time
from typing import Optional

import numpy as np

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, SportModeState_

from state.leg_dof_mapping import motor_legs_to_sim
from utils.math_utils import quat_rotate_inverse_np


class B2StateReader:
    """Subscribe to B2 rt/lowstate and sport mode state."""

    def __init__(self, interface: str, sport_topic: str, sport_topic_fallback: str):
        self.interface = interface
        self.sport_topic = sport_topic
        self.sport_topic_fallback = sport_topic_fallback
        self._lock = threading.Lock()
        self._low_state: Optional[LowState_] = None
        self._sport_state: Optional[SportModeState_] = None
        self._initialized = False

    def release_motion_mode(self) -> None:
        """Release sport mode so rt/lowcmd can take over."""
        self._release_motion_mode()

    def initialize(self, release_motion_mode: bool = True) -> None:
        ChannelFactoryInitialize(0, self.interface)

        self._low_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._low_sub.Init(self._on_low_state, 10)

        self._sport_sub = ChannelSubscriber(self.sport_topic, SportModeState_)
        self._sport_sub.Init(self._on_sport_state, 10)
        self._active_sport_topic = self.sport_topic

        if release_motion_mode:
            self._release_motion_mode()

        deadline = time.time() + 5.0
        while time.time() < deadline:
            with self._lock:
                if self._low_state is not None:
                    self._initialized = True
                    return
            time.sleep(0.05)
        raise TimeoutError("Timed out waiting for B2 rt/lowstate")

    def _release_motion_mode(self) -> None:
        """Release sport mode for rt/lowcmd (ReleaseMode only; do not StandDown)."""
        msc = MotionSwitcherClient()
        msc.SetTimeout(5.0)
        msc.Init()
        deadline = time.time() + 30.0
        while time.time() < deadline:
            status, result = msc.CheckMode()
            if not result.get("name"):
                return
            msc.ReleaseMode()
            time.sleep(0.5)
        raise TimeoutError("Timed out releasing B2 motion mode (sport controller still active)")

    def _on_low_state(self, msg: LowState_) -> None:
        with self._lock:
            self._low_state = msg

    def _on_sport_state(self, msg: SportModeState_) -> None:
        with self._lock:
            self._sport_state = msg

    def read(self) -> dict:
        with self._lock:
            low = self._low_state
            sport = self._sport_state

        if low is None:
            return {"valid": False}

        imu = low.imu_state
        base_quat = np.array(imu.quaternion, dtype=np.float64)
        if base_quat[0] == 0.0 and np.allclose(base_quat[1:], 0.0):
            base_quat = np.array([1.0, 0.0, 0.0, 0.0])

        gyro = np.array(imu.gyroscope, dtype=np.float64)
        leg_q_motor = np.array([low.motor_state[i].q for i in range(12)], dtype=np.float64)
        leg_dq_motor = np.array([low.motor_state[i].dq for i in range(12)], dtype=np.float64)
        leg_q = motor_legs_to_sim(leg_q_motor)
        leg_dq = motor_legs_to_sim(leg_dq_motor)
        # Raw motor-order q kept for diagnostics (leg DOF mapping verification).
        leg_q_motor_raw = leg_q_motor.copy()
        foot_force = np.array(low.foot_force, dtype=np.float64)

        base_pos = np.zeros(3, dtype=np.float64)
        base_lin_vel_world = np.zeros(3, dtype=np.float64)
        # Intentionally ignore sport.position: deploy kinematics use the robot
        # base frame (origin at base). Sport velocity is still used for twist.
        if sport is not None:
            base_lin_vel_world = np.array(sport.velocity, dtype=np.float64)
            if abs(sport.yaw_speed) > 1e-8:
                gyro[2] = float(sport.yaw_speed)

        base_lin_vel_local = quat_rotate_inverse_np(base_quat, base_lin_vel_world)
        base_ang_vel_local = quat_rotate_inverse_np(base_quat, gyro)
        projected_gravity = quat_rotate_inverse_np(base_quat, np.array([0.0, 0.0, -1.0]))

        return {
            "valid": True,
            "base_quat": base_quat,
            "base_pos": base_pos,
            "base_lin_vel_world": base_lin_vel_world,
            "base_ang_vel_world": gyro,
            "base_lin_vel_local": base_lin_vel_local,
            "base_ang_vel_local": base_ang_vel_local,
            "leg_q": leg_q,
            "leg_dq": leg_dq,
            "leg_q_motor": leg_q_motor_raw,
            "foot_force": foot_force,
            "projected_gravity": projected_gravity,
            "timestamp": time.time(),
        }

    @property
    def sport_topic_in_use(self) -> str:
        return getattr(self, "_active_sport_topic", self.sport_topic)
