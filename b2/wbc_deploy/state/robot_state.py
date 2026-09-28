"""Unified robot state for B2Z1 WBC deploy."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class RobotState:
    """Snapshot of B2 base + legs and Z1 arm at one timestep."""

    base_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    # Unused for kinematics (always zeros). Kept for API compatibility.
    base_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    base_lin_vel_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    base_ang_vel_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    base_lin_vel_local: np.ndarray = field(default_factory=lambda: np.zeros(3))
    base_ang_vel_local: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # URDF/sim order: FL, FR, RL, RR (remapped from B2 motor order in b2_reader).
    leg_q: np.ndarray = field(default_factory=lambda: np.zeros(12))
    leg_dq: np.ndarray = field(default_factory=lambda: np.zeros(12))
    leg_q_motor: Optional[np.ndarray] = None
    arm_q: np.ndarray = field(default_factory=lambda: np.zeros(6))
    arm_dq: np.ndarray = field(default_factory=lambda: np.zeros(6))
    gripper_q: float = 0.0
    # End-effector pose in **robot base frame** (not sport-world odom).
    ee_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ee_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    foot_force: np.ndarray = field(default_factory=lambda: np.zeros(4))
    foot_contacts: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=bool))
    projected_gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -1.0]))
    jacobian: Optional[np.ndarray] = None
    arm_connected: bool = False
    valid: bool = False
    timestamp: float = 0.0

    @property
    def qpos(self) -> np.ndarray:
        return np.concatenate([self.leg_q, self.arm_q, np.array([self.gripper_q])])

    @property
    def qvel(self) -> np.ndarray:
        return np.concatenate([self.leg_dq, self.arm_dq, np.array([0.0])])
