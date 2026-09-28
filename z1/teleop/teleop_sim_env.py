"""Factory for b2z1 VR teleoperation MuJoCo environment."""

import os

from constants import DT, XML_DIR
from dm_control import mujoco
from dm_control.rl import control

from teleop.teleop_task import TeleopTask


def make_teleop_sim_env(time_limit=300):
    xml_path = os.path.join(XML_DIR, "b2z1/scene_teleop.xml")
    physics = mujoco.Physics.from_xml_path(xml_path)
    task = TeleopTask()
    env = control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=DT,
        n_sub_steps=None,
        flat_observation=False,
    )
    return env
