"""Minimal B2 DDS reader for base orientation + linear velocity (no ReleaseMode)."""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np


def _quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_w = q[0]
    q_vec = q[1:]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


class B2DdsReader:
    """Subscribe to rt/lowstate (+ sport) for quat and local linear velocity."""

    def __init__(
        self,
        interface: str = "enp8s0",
        sport_topic: str = "rt/lf/sportmodestate",
    ):
        self.interface = interface
        self.sport_topic = sport_topic
        self._lock = threading.Lock()
        self._low_state = None
        self._sport_state = None
        self._initialized = False

    def start(self) -> None:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, SportModeState_

        ChannelFactoryInitialize(0, self.interface)

        self._low_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._low_sub.Init(self._on_low_state, 10)

        self._sport_sub = ChannelSubscriber(self.sport_topic, SportModeState_)
        self._sport_sub.Init(self._on_sport_state, 10)

        deadline = time.time() + 5.0
        while time.time() < deadline:
            with self._lock:
                if self._low_state is not None:
                    self._initialized = True
                    print(
                        f"[B2DdsReader] connected on {self.interface} "
                        f"(sport={self.sport_topic})",
                        flush=True,
                    )
                    return
            time.sleep(0.05)
        raise TimeoutError(
            f"Timed out waiting for B2 rt/lowstate on interface {self.interface}"
        )

    def _on_low_state(self, msg) -> None:
        with self._lock:
            self._low_state = msg

    def _on_sport_state(self, msg) -> None:
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
            base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        base_lin_vel_world = np.zeros(3, dtype=np.float64)
        if sport is not None:
            base_lin_vel_world = np.array(sport.velocity, dtype=np.float64)

        base_lin_vel_local = _quat_rotate_inverse(base_quat, base_lin_vel_world)
        return {
            "valid": True,
            "base_quat": base_quat,
            "base_lin_vel_local": base_lin_vel_local,
            "base_lin_vel_world": base_lin_vel_world,
            "timestamp": time.time(),
        }
