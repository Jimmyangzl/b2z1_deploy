"""B2 motor index order vs Isaac URDF / policy joint order for leg DOFs.

B2 ``rt/lowstate`` motors 0–11: FR, FL, RR, RL (hip/thigh/calf each).
Isaac Gym, URDF, policy, and ``default_joint_angles``: FL, FR, RL, RR.

Foot force on B2 is already FL, FR, RL, RR and needs no remap.

Training reference (``reorder_dofs = False`` in b2z1_lambdawbc_config.py): policy
observations use Isaac order; only B2 hardware reads/writes need this permutation.
"""

import numpy as np

# Policy / URDF DOF order (matches Isaac ``get_asset_dof_names`` for b2z1.urdf).
SIM_LEG_DOF_NAMES = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)

SIM_DOF_NAMES = SIM_LEG_DOF_NAMES + (
    "z1_waist",
    "z1_shoulder",
    "z1_elbow",
    "z1_wrist_angle",
    "z1_forearm_roll",
    "z1_wrist_rotate",
    "z1_jointGripper",
)

# B2 low-level motor order (unitree_legged_const LegID).
MOTOR_LEG_DOF_NAMES = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
)

# sim joint index s -> B2 motor index
SIM_TO_MOTOR = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=np.intp)
# B2 motor index m -> sim joint index
MOTOR_TO_SIM = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=np.intp)


def motor_legs_to_sim(values: np.ndarray) -> np.ndarray:
    """Reorder 12 leg values from B2 motor order to URDF/sim order."""
    motor = np.asarray(values, dtype=np.float64).reshape(12)
    sim = np.empty(12, dtype=np.float64)
    for s in range(12):
        sim[s] = motor[SIM_TO_MOTOR[s]]
    return sim


def sim_legs_to_motor(values: np.ndarray) -> np.ndarray:
    """Reorder 12 leg values from URDF/sim order to B2 motor order."""
    sim = np.asarray(values, dtype=np.float64).reshape(12)
    motor = np.empty(12, dtype=np.float64)
    for m in range(12):
        motor[m] = sim[MOTOR_TO_SIM[m]]
    return motor


def default_dof_pos_from_config(joint_angles: dict) -> np.ndarray:
    """Build default DOF vector in sim/policy joint order."""
    return np.array([float(joint_angles[name]) for name in SIM_DOF_NAMES], dtype=np.float64)


def leg_mapping_diagnostic(leg_q_motor: np.ndarray, leg_q_sim: np.ndarray) -> dict:
    """Compare motor vs remapped sim leg q; useful to verify FR/FL are not swapped."""
    motor = np.asarray(leg_q_motor, dtype=np.float64).reshape(12)
    sim = np.asarray(leg_q_sim, dtype=np.float64).reshape(12)
    unmapped = motor.copy()  # treating motor order as sim would
    return {
        "motor_FR_hip": float(motor[0]),
        "motor_FL_hip": float(motor[3]),
        "sim_FL_hip": float(sim[0]),
        "sim_FR_hip": float(sim[3]),
        "swap_detected": abs(sim[0] - motor[0]) < abs(sim[0] - motor[3]),
    }
