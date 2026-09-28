"""Non-ROS Vive tracker reader adapted from vive_vr_tracking_ros2."""

import math
import os
import sys
import time

import numpy as np
import yaml

def _import_vive_tracker_module():
    """Import OpenVR tracker code only when reading hardware.

    The Z1 websocket client only needs load_config(). Importing ViveTrackerModule
    at module load time would fail on machines that do not have vr_tracking.
    """
    teleop_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(teleop_dir, "vive_vr_tracking_ros2"),
        os.path.join(teleop_dir, "../../../vive_vr_tracking_ros2"),
    ]
    _VIVE_PKG = None
    for candidate in candidates:
        path = os.path.abspath(candidate)
        if os.path.isdir(path):
            _VIVE_PKG = path
            break
    if _VIVE_PKG is None:
        raise ModuleNotFoundError(
            "Could not find vive_vr_tracking_ros2. Expected it under teleop/vive_vr_tracking_ros2."
        )
    if _VIVE_PKG not in sys.path:
        sys.path.insert(0, _VIVE_PKG)
    from vr_tracking.track import ViveTrackerModule  # noqa: E402

    return ViveTrackerModule


def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotation_matrix_to_quaternion(R):
    assert R.shape == (3, 3), "Rotation matrix must be 3x3"
    trace = np.trace(R)
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    return np.array([w, x, y, z])


class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0, freq=50.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.freq = freq
        self.x_prev = None
        self.dx_prev = None
        self.alpha = self.compute_alpha(min_cutoff)

    def compute_alpha(self, cutoff):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau * self.freq)

    def filter(self, x):
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            return x
        dx = (x - self.x_prev) * self.freq
        self.dx_prev = self.dx_prev + self.compute_alpha(self.d_cutoff) * (dx - self.dx_prev)
        cutoff = self.min_cutoff + self.beta * abs(self.dx_prev)
        self.alpha = self.compute_alpha(cutoff)
        x_hat = self.x_prev + self.alpha * (x - self.x_prev)
        self.x_prev = x_hat
        return x_hat


class VRTracker:
    def __init__(self, tracker):
        self.tracker = tracker
        self.o_R_b = None
        self.origin = None
        self.calibrate_flag = False
        self.position_filter = [
            OneEuroFilter(min_cutoff=1.0, beta=0.01, d_cutoff=1.0, freq=50.0)
            for _ in range(3)
        ]

    def frame_calibrate(self):
        try:
            b_pose_t = np.array(list(self.tracker.get_pose_matrix()))
            b_R_t = b_pose_t[:3, :3]
            # self.o_R_b = np.array([b_R_t[:, 0], -b_R_t[:, 1], b_R_t[:, 2]])
            self.o_R_b = b_R_t
            self.origin = self.o_R_b @ b_pose_t[:3, -1]
            self.calibrate_flag = True
        except Exception:
            self.o_R_b = None
            self.origin = None
            self.calibrate_flag = False


class LeftTrackerReader:
    """Reads calibrated left Vive tracker pose without ROS."""

    def __init__(self, config_path=None):
        self.config = load_config(config_path)
        ViveTrackerModule = _import_vive_tracker_module()
        self.vive_tracker = ViveTrackerModule()
        self.vive_tracker.print_discovered_objects()
        # self.R_t = np.array([
        #     [0.0, 1.0, 0.0],
        #     [-1.0, 0.0, 0.0],
        #     [0.0, 0.0, 1.0],
        # ])
        self.R_t = np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ])
        # z_rot = float(self.config.get("tracker_z_rotation", 0.0))
        # self.R_z_align = rotation_z(z_rot)
        self.tracker = None
        self._find_left_tracker()

    def _find_left_tracker(self):
        serial = self.config["left_tracker_serial"]
        for index, value in self.vive_tracker.devices.items():
            if "tracker" in index and value.get_serial() == serial:
                self.tracker = VRTracker(value)
                return
        raise RuntimeError(f"No left tracker found with serial {serial}")

    def calibrate(self):
        wait_time = float(self.config.get("frame_cali_wait_time", 2.0))
        print(f"Frame calibration starts in {wait_time}s, please hold the tracker still.")
        time.sleep(wait_time)
        self.tracker.frame_calibrate()
        if not self.tracker.calibrate_flag:
            raise RuntimeError("Left tracker frame calibration failed.")
        print("Left tracker frame calibrated.")

    def get_pose(self):
        """Return dict with connected, position (3,), orientation (4,) [w,x,y,z]."""
        if self.tracker is None or not self.tracker.calibrate_flag:
            return {"connected": False, "position": None, "orientation": None}

        pose_matrix = self.tracker.tracker.get_pose_matrix()
        if pose_matrix is None:
            return {"connected": False, "position": None, "orientation": None}

        pose_b = np.array(list(pose_matrix))
        pose_b[:3, :3] = pose_b[:3, :3] @ self.R_t
        pose_o = self.tracker.o_R_b @ pose_b
        pose_o[:3, -1] = pose_o[:3, -1] - self.tracker.origin
        # Extra alignment: rotate calibrated frame by tracker_z_rotation about local z.
        # pose_o[:3, :3] = pose_o[:3, :3] @ self.R_z_align
        # pose_o[:3, -1] = self.R_z_align @ pose_o[:3, -1]
        gain = float(self.config.get("translational_gain", 2.0))
        for i in range(3):
            pose_o[i, -1] = self.tracker.position_filter[i].filter(pose_o[i, -1] * gain)

        position = pose_o[:3, -1].astype(np.float64)
        orientation = rotation_matrix_to_quaternion(pose_o[:3, :3].astype(np.float64))
        return {
            "connected": True,
            "position": position,
            "orientation": orientation,
        }
