"""Aggregate B2 and Z1 readings into a single RobotState."""

import math
import os
import time

import numpy as np
import yaml

from state.b2_reader import B2StateReader
from state.robot_state import RobotState
from state.z1_reader import Z1StateReader
from utils.urdf_arm_base import load_arm_base_offset_from_urdf

# waist, shoulder, elbow, wrist_angle, forearm_roll, wrist_rotate
_Z1_DEFAULT_ANGLE_KEYS = (
    "z1_waist",
    "z1_shoulder",
    "z1_elbow",
    "z1_wrist_angle",
    "z1_forearm_roll",
    "z1_wrist_rotate",
)
_DEFAULT_ARM_INIT_JOINT = [0.0, 0.6, -0.6, 0.0, 0.0, math.pi / 2.0]


def apply_arm_init_joint(config: dict) -> list:
    """Normalize arm_init_joint and sync arm_home_joint + default_joint_angles z1_*."""
    raw = config.get("arm_init_joint", config.get("arm_home_joint", _DEFAULT_ARM_INIT_JOINT))
    arm_init = [float(x) for x in raw]
    if len(arm_init) != 6:
        raise ValueError(f"arm_init_joint must have 6 values, got {len(arm_init)}")

    config["arm_init_joint"] = arm_init
    config["arm_home_joint"] = list(arm_init)

    robot = config.setdefault("robot", {})
    angles = robot.setdefault("default_joint_angles", {})
    for key, value in zip(_Z1_DEFAULT_ANGLE_KEYS, arm_init):
        angles[key] = value
    return arm_init


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    robot = config.setdefault("robot", {})
    urdf_rel = robot.get("urdf", "urdf/b2z1.urdf")
    config_dir = os.path.dirname(os.path.abspath(config_path))
    urdf_path = urdf_rel if os.path.isabs(urdf_rel) else os.path.join(config_dir, urdf_rel)
    arm_base = load_arm_base_offset_from_urdf(urdf_path)
    robot["urdf"] = urdf_path
    robot["policy_arm_base_offset"] = arm_base
    robot["z1_base_in_robot_base"] = list(arm_base)
    print(f"Arm mount from URDF base_static_joint: {arm_base}  ({urdf_path})")

    arm_init = apply_arm_init_joint(config)
    print(f"Arm init joint (home / defaults / teleop prep): {arm_init}")
    return config


class RobotStateAggregator:
    """Combine B2 DDS state and Z1 SDK state."""

    def __init__(self, config: dict):
        net = config["network"]
        robot = config["robot"]
        obs_cfg = config["observation"]
        self.foot_contact_threshold = float(obs_cfg["foot_contact_threshold"])

        self.b2 = B2StateReader(
            interface=net["interface"],
            sport_topic=net["sport_topic"],
            sport_topic_fallback=net["sport_topic_fallback"],
        )
        self.z1 = Z1StateReader(
            z1_base_in_robot_base=np.array(robot["z1_base_in_robot_base"], dtype=np.float64),
            default_arm_q=np.array(config["arm_init_joint"], dtype=np.float64),
        )

    def release_motion_mode(self) -> None:
        self.b2.release_motion_mode()

    def initialize(
        self,
        release_motion_mode: bool = True,
        start_z1_loop: bool = True,
    ) -> None:
        self.b2.initialize(release_motion_mode=release_motion_mode)
        self.z1.initialize(start_loop=start_z1_loop)

    def start_z1_loop(self) -> None:
        self.z1.start_loop()

    def close(self) -> None:
        self.z1.close()

    def read(self) -> RobotState:
        b2 = self.b2.read()
        if not b2["valid"]:
            return RobotState(valid=False)

        z1 = self.z1.read()

        foot_contacts = b2["foot_force"] > self.foot_contact_threshold

        return RobotState(
            base_quat=b2["base_quat"],
            # Always zero: EE / goals are in robot base frame (no sport odom).
            base_pos=np.zeros(3, dtype=np.float64),
            base_lin_vel_world=b2["base_lin_vel_world"],
            base_ang_vel_world=b2["base_ang_vel_world"],
            base_lin_vel_local=b2["base_lin_vel_local"],
            base_ang_vel_local=b2["base_ang_vel_local"],
            leg_q=b2["leg_q"],
            leg_dq=b2["leg_dq"],
            leg_q_motor=b2.get("leg_q_motor"),
            arm_q=z1["arm_q"],
            arm_dq=z1["arm_dq"],
            gripper_q=z1["gripper_q"],
            ee_pos=z1["ee_pos"],
            ee_quat=z1["ee_quat"],
            foot_force=b2["foot_force"],
            foot_contacts=foot_contacts,
            projected_gravity=b2["projected_gravity"],
            jacobian=z1["jacobian"],
            arm_connected=z1["arm_connected"],
            valid=True,
            timestamp=time.time(),
        )
