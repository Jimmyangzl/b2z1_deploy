#!/usr/bin/env python3
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from paths import setup_paths

setup_paths()

from obs.process_obs_worker import ProcessObsWorker
from policy.constants import num_observations
from state.aggregator import load_config
from state.robot_state import RobotState


def main():
    cfg = load_config(os.path.join(ROOT, "config/b2z1_wbc.yaml"))
    w = ProcessObsWorker(cfg, control_dt=0.02, use_vr=False)
    w.start()
    w.wait_ready(60.0)
    robot = RobotState(valid=True, arm_connected=True)
    robot.jacobian = np.eye(6)
    pack = None
    for i in range(80):
        w.submit(robot, None, debug=(i == 0))
        time.sleep(0.01)
        got = w.take()
        if got is not None:
            pack = got
    assert pack is not None, "no obs returned"
    assert pack["obs_vector"].shape == (num_observations,), pack["obs_vector"].shape
    print("build_ms", w.last_build_ms, "obs", pack["obs_vector"].shape)
    w.stop()
    print("OK")


if __name__ == "__main__":
    main()
