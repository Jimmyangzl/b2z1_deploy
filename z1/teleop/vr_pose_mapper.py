"""Map Vive tracker pose to end-effector goal with canonical base frame alignment."""

import os

import numpy as np
import yaml

from helperfunc.helper_funcs import quat_to_rotmat


def _load_mapper_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def _normalize_quat(quat):
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return quat / norm


def _rotation_matrix_to_quaternion(R):
    trace = np.trace(R)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return _normalize_quat(np.array([w, x, y, z]))


def make_canonical_rotation(base_yaw):
    """EE base frame: x forward, y up, z lateral; x-z plane parallel to ground."""
    forward = np.array([np.cos(base_yaw), np.sin(base_yaw), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    lateral = np.cross(forward, up)
    return np.column_stack([forward, up, lateral])


def make_canonical_quat(base_yaw):
    return _rotation_matrix_to_quaternion(make_canonical_rotation(base_yaw))


def _load_body_to_ee_matrix(config):
    matrix = config.get("vr_body_to_ee_matrix")
    if matrix is not None:
        return np.asarray(matrix, dtype=np.float64)
    return np.eye(3, dtype=np.float64)


def _rotation_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rotation_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ee_base_delta_rotation(roll_x, pitch_z, yaw_y):
    """Build delta rotation in EE base frame: roll-x, pitch-z, yaw-y."""
    return _rotation_y(yaw_y) @ _rotation_z(pitch_z) @ _rotation_x(roll_x)


def _rotation_matrix_to_rotvec(R):
    cos_angle = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    if angle < 1e-8:
        return np.zeros(3, dtype=np.float64)
    axis = np.array(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ],
        dtype=np.float64,
    )
    axis = axis / (2.0 * np.sin(angle))
    return axis * angle


def _remap_body_rotation(R_delta_body, R_body_to_ee):
    """Map tracker-body delta rotation into EE-base roll/pitch/yaw.

    Uses the same axis permutation as translation: split body rotvec into
    x/y/z components, remap with R_body_to_ee, then compose independent EE
    rotations about x (roll), z (pitch), and y (yaw).
    """
    rotvec_body = _rotation_matrix_to_rotvec(R_delta_body)
    mapped = R_body_to_ee @ rotvec_body
    roll_x = -mapped[2]
    pitch_z = mapped[0]
    yaw_y = mapped[1]
    return _ee_base_delta_rotation(roll_x, pitch_z, yaw_y)


class VrPoseMapper:
    """Convert VR tracker messages into gravity-aligned EE goals (base origin).

    Callers that work in the robot base frame should rotate EE into this
    aligned frame before ``update`` and rotate the returned goal back
    (see ``VrGoalProvider``).
    """

    MODE_IDLE = "idle"
    MODE_ACTIVE = "active"

    def __init__(self, config_path=None):
        self.config_path = config_path
        self.config = _load_mapper_config(config_path)
        self.R_body_to_ee = _load_body_to_ee_matrix(self.config)
        self.mode = self.MODE_IDLE
        self.goal_pos = None
        self.goal_quat = None
        self._T_ee_base = None
        self._p_vr0 = None
        self._R_vr0 = None

    def reset(self):
        self.mode = self.MODE_IDLE
        self.goal_pos = None
        self.goal_quat = None
        self._T_ee_base = None
        self._p_vr0 = None
        self._R_vr0 = None

    def _make_ee_base_matrix(self, ee_pos, base_yaw):
        R_ee = make_canonical_rotation(base_yaw)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_ee
        T[:3, 3] = np.asarray(ee_pos, dtype=np.float64)
        return T

    def update(self, vr_pose, ee_pos, ee_quat, base_yaw):
        ee_pos = np.asarray(ee_pos, dtype=np.float64).copy()
        canonical_quat = make_canonical_quat(base_yaw)

        if not vr_pose.get("connected", False):
            self.mode = self.MODE_IDLE
            self._T_ee_base = None
            self._p_vr0 = None
            self._R_vr0 = None
            self.goal_pos = ee_pos.copy()
            self.goal_quat = canonical_quat.copy()
            return self.goal_pos, self.goal_quat

        p_vr = np.asarray(vr_pose["position"], dtype=np.float64)
        R_vr = quat_to_rotmat(_normalize_quat(vr_pose["orientation"]))

        if self.mode == self.MODE_IDLE or self._T_ee_base is None:
            self.mode = self.MODE_ACTIVE
            self._T_ee_base = self._make_ee_base_matrix(ee_pos, base_yaw)
            self._p_vr0 = p_vr.copy()
            self._R_vr0 = R_vr.copy()

        R_ee = self._T_ee_base[:3, :3]
        p_ee = self._T_ee_base[:3, 3]

        dp_body = self._R_vr0.T @ (p_vr - self._p_vr0)
        R_delta_body = self._R_vr0.T @ R_vr

        dp_ee = self.R_body_to_ee @ dp_body
        R_delta_ee = _remap_body_rotation(R_delta_body, self.R_body_to_ee)

        self.goal_pos = p_ee + R_ee @ dp_ee
        self.goal_quat = _rotation_matrix_to_quaternion(R_ee @ R_delta_ee)
        return self.goal_pos.copy(), self.goal_quat.copy()

    def get_goal(self):
        if self.goal_pos is None or self.goal_quat is None:
            return None, None
        return self.goal_pos.copy(), self.goal_quat.copy()
