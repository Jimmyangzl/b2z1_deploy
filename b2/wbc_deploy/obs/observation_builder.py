"""Build WBC observation vector matching teleop_task / controller_loader layout."""

from typing import Any, Dict, TYPE_CHECKING

import math

import numpy as np

from obs.gait_clock import GaitClock
from obs.manipulability import ManipulabilityTracker
from policy.constants import history_len, num_priv, num_proprio
from state.arm_defaults import arm_state_valid
from state.leg_dof_mapping import default_dof_pos_from_config
from state.robot_state import RobotState
from vr_goal.vr_goal_provider import NullVrGoalProvider
from utils.math_utils import (
    quat_apply,
    quat_mul,
    quat_rotate_inverse_np,
    quaternion_to_rpy,
    rpy_to_quaternion,
)

if TYPE_CHECKING:
    from vr_goal.vr_goal_provider import VrGoalProvider


class ObservationBuilder:
    """Construct 953-dim WBC observation from robot state and VR goal."""

    def __init__(self, config: dict, control_dt: float):
        self.config = config
        obs_cfg = config["observation"]
        robot_cfg = config["robot"]
        cmd_cfg = config.get("commands", {})
        self.lin_vel_scale = float(obs_cfg["lin_vel_scale"])
        self.ang_vel_scale = float(obs_cfg["ang_vel_scale"])
        self.dof_vel_scale = float(obs_cfg["dof_vel_scale"])
        self.z1_base_in_robot_base = np.array(robot_cfg["z1_base_in_robot_base"], dtype=np.float64)
        policy_arm_base = robot_cfg.get("policy_arm_base_offset", [0.15, 0.0, 0.10])
        self.policy_arm_base_offset = np.array(policy_arm_base, dtype=np.float64)
        # When True, last_leg_actions in proprio stay zero (breaks action feedback; use for --no-vr standing test).
        self.suppress_last_leg_actions = False

        self.use_cmd = bool(cmd_cfg.get("use_cmd", False))
        self.commands = np.array(
            [
                float(cmd_cfg.get("cmd_vx", 0.0)),
                float(cmd_cfg.get("cmd_vy", 0.0)),
                float(cmd_cfg.get("cmd_yaw", 0.0)),
            ],
            dtype=np.float64,
        )
        self.commands_scale = np.array(
            [self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale],
            dtype=np.float64,
        )
        self.lin_vel_x_clip = float(cmd_cfg.get("lin_vel_x_clip", 0.1))
        self.lin_vel_y_clip = float(cmd_cfg.get("lin_vel_y_clip", 0.1))
        self.ang_vel_yaw_clip = float(cmd_cfg.get("ang_vel_yaw_clip", 0.05))

        angles = robot_cfg["default_joint_angles"]
        self.default_dof_pos = default_dof_pos_from_config(angles)

        self.gait_clock = GaitClock(dt=control_dt, cfg=config["gait"])
        self.manip = ManipulabilityTracker()
        self.obs_history_buf = np.zeros((history_len, num_proprio), dtype=np.float64)
        self.action_history_buf = np.zeros((5, 12), dtype=np.float64)
        self.episode_length_buf = -1
        self.dvel_b_local = np.zeros(3, dtype=np.float64)
        self.dpos_b_local = np.zeros(3, dtype=np.float64)
        self.phase_variable = 0.0
        self.privileged_obs = np.zeros(num_priv, dtype=np.float64)

    def set_commands(self, vx: float, vy: float, yaw: float) -> None:
        """Update base velocity command used when use_cmd is true."""
        self.commands[0] = float(vx)
        self.commands[1] = float(vy)
        self.commands[2] = float(yaw)
        self.use_cmd = True

    def reset(self) -> None:
        self.obs_history_buf[:] = 0.0
        self.action_history_buf[:] = 0.0
        self.episode_length_buf = -1
        self.dvel_b_local[:] = 0.0
        self.dpos_b_local[:] = 0.0
        self.phase_variable = 0.0
        self.gait_clock.gait_indices[:] = 0.0
        self.gait_clock.clock_inputs[:] = 0.0
        self.manip = ManipulabilityTracker()

    def update_action_history(self, action: np.ndarray) -> None:
        leg = np.asarray(action, dtype=np.float64).reshape(-1)[:12]
        self.action_history_buf = np.vstack([self.action_history_buf[1:], leg.reshape(1, -1)])

    @staticmethod
    def get_body_rp(base_quat: np.ndarray) -> np.ndarray:
        return quaternion_to_rpy(base_quat)[:2]

    def _base_to_base_yaw_pos(self, pos_base: np.ndarray, base_quat: np.ndarray) -> np.ndarray:
        """Robot-base position → gravity-aligned yaw frame (origin at base)."""
        base_yaw = quaternion_to_rpy(base_quat)[2]
        base_yaw_quat = rpy_to_quaternion(np.array([0.0, 0.0, base_yaw]))
        pos_aligned = quat_apply(base_quat, pos_base)
        return quat_rotate_inverse_np(base_yaw_quat, pos_aligned)

    def _ee_goal_local_cart(self, goal_base: np.ndarray) -> np.ndarray:
        """EE goal in arm-base frame (mount origin; axes = B2/arm, rpy=0 mount)."""
        return np.asarray(goal_base, dtype=np.float64) - self.policy_arm_base_offset

    def _update_rfm_state(
        self,
        robot: RobotState,
        vr_goal: "VrGoalProvider",
        manip_pred_last: float,
    ) -> None:
        """Build phase_variable and dvel_b_local to match Isaac arm_control RFM.

        dvel (when VR active):
          xy = 0.2 * unit(goal in base_yaw xy)
          yaw = 0.4 * sign(goal_yaw - base_yaw)
        masks (same as arm_control.update_rfm_state):
          mask_xy:   manip_pred > 1e-4 and phase < 0.5  → zero xy
          mask_back: manip_pred < 1e-3 and dist(ee, mount)_xy < 0.4 and phase < 0.5
                     → flip xy
          mask_yaw:  |yaw_err| > 0.785  → keep yaw; else zero yaw
        """
        goal_pos = vr_goal.get_goal_position(robot.ee_pos)  # robot base frame
        base_quat = np.asarray(robot.base_quat, dtype=np.float64)
        base_yaw = float(quaternion_to_rpy(base_quat)[2])

        # Horizontal EE↔goal distance in gravity-aligned yaw frame (phase).
        ee_yaw = self._base_to_base_yaw_pos(robot.ee_pos, base_quat)
        goal_yaw_pos = self._base_to_base_yaw_pos(goal_pos, base_quat)
        dpos_xy_norm = float(np.linalg.norm((goal_yaw_pos - ee_yaw)[:2]))
        self.phase_variable = 1.0 / (1.0 + np.exp(-5.0 * (dpos_xy_norm - 1.0) / 1.0))

        # Goal relative to base origin in base_yaw (sim: R_yaw^{-1}(goal_world - base_pos)).
        self.dpos_b_local = goal_yaw_pos.copy()
        dvel_norm = float(np.linalg.norm(self.dpos_b_local[:2]))
        if dvel_norm > 1e-8:
            self.dvel_b_local[:2] = 0.2 * self.dpos_b_local[:2] / dvel_norm
        else:
            self.dvel_b_local[:2] = 0.0

        goal_quat_base = getattr(vr_goal, "ee_quat_goal_base", None)
        if goal_quat_base is None:
            goal_quat_base = getattr(vr_goal, "ee_quat_goal_world", None)
        if goal_quat_base is None:
            goal_quat_base = robot.ee_quat
        goal_quat_base = np.asarray(goal_quat_base, dtype=np.float64)
        # Align goal quat to gravity frame (inverse of VrGoalProvider base conversion).
        goal_quat_aligned = quat_mul(base_quat, goal_quat_base)
        ee_goal_yaw = float(quaternion_to_rpy(goal_quat_aligned)[2])
        yaw_diff = math.atan2(
            math.sin(ee_goal_yaw - base_yaw),
            math.cos(ee_goal_yaw - base_yaw),
        )
        yaw_direction = float(np.sign(yaw_diff)) if abs(yaw_diff) > 1e-8 else 0.0
        self.dvel_b_local[2] = 0.4 * yaw_direction
        ee_euler_error = abs(yaw_diff)

        mask_xy = (manip_pred_last > 0.0001) and (self.phase_variable < 0.5)
        mask_yaw = ee_euler_error > 0.785

        arm_mount_yaw = self._base_to_base_yaw_pos(self.policy_arm_base_offset, base_quat)
        dist_base_ee = float(np.linalg.norm((ee_yaw - arm_mount_yaw)[:2]))
        mask_back = (
            (manip_pred_last < 0.001)
            and (dist_base_ee < 0.4)
            and (self.phase_variable < 0.5)
        )

        if mask_xy:
            self.dvel_b_local[:2] = 0.0
        if mask_back:
            self.dvel_b_local[:2] = -self.dvel_b_local[:2]
        if not mask_yaw:
            self.dvel_b_local[2] = 0.0

        if not vr_goal.is_active:
            self.dvel_b_local[:] = 0.0

    def _locomotion_obs(self) -> np.ndarray:
        if self.use_cmd:
            return self.commands * self.commands_scale
        return self.dvel_b_local.copy()

    def _gait_locomotion_cmd(self) -> np.ndarray:
        if self.use_cmd:
            return self.commands.copy()
        return self.dvel_b_local.copy()

    def _obs_joint_state(self, robot: RobotState, vr_goal: "VrGoalProvider"):
        """Joint state for proprio; reject corrupt arm SDK values."""
        qpos = robot.qpos.copy()
        qvel = robot.qvel.copy()
        arm_ok = arm_state_valid(qpos[12:18], qvel[12:18])
        if not arm_ok or (isinstance(vr_goal, NullVrGoalProvider) and not robot.arm_connected):
            qpos[12:18] = self.default_dof_pos[12:18]
            qvel[12:18] = 0.0
        return qpos, qvel

    def build(
        self,
        robot: RobotState,
        vr_goal: "VrGoalProvider",
    ) -> Dict[str, Any]:
        self.episode_length_buf += 1

        vr_goal.update(robot.ee_pos, robot.ee_quat, robot.base_quat)
        goal_base = vr_goal.get_goal_position(robot.ee_pos)

        if isinstance(vr_goal, NullVrGoalProvider):
            manip_det = 0.0007
            manip_pred_last = 0.0007
        else:
            manip_det = self.manip.update(robot.jacobian)
            manip_pred_last = float(self.manip.manip_det_pred[-1])
        self._update_rfm_state(robot, vr_goal, manip_pred_last)

        ee_goal_local_cart = self._ee_goal_local_cart(goal_base)
        # ee_goal_local_cart[0] = 0.4 
        # ee_goal_local_cart[2] = 0.4 

        locomotion_obs = self._locomotion_obs()
        gait_indices, clock_inputs = self.gait_clock.step(
            self._gait_locomotion_cmd(),
            use_cmd=self.use_cmd,
            lin_vel_x_clip=self.lin_vel_x_clip,
            lin_vel_y_clip=self.lin_vel_y_clip,
            ang_vel_yaw_clip=self.ang_vel_yaw_clip,
        )

        qpos, qvel = self._obs_joint_state(robot, vr_goal)
        body_rp = self.get_body_rp(robot.base_quat)
        base_lin_vel = robot.base_lin_vel_local * self.lin_vel_scale
        base_ang_vel = robot.base_ang_vel_local * self.ang_vel_scale
        dof_pos_err = (qpos - self.default_dof_pos)[:-1]
        dof_vel_scaled = qvel[:-1] * self.dof_vel_scale
        if self.suppress_last_leg_actions:
            last_leg_actions = np.zeros(12, dtype=np.float64)
        else:
            last_leg_actions = self.action_history_buf[-1, :12]
        foot_contacts = robot.foot_contacts.astype(np.float64)

        proprio = np.concatenate(
            [
                body_rp,
                base_lin_vel,
                base_ang_vel,
                dof_pos_err,
                dof_vel_scaled,
                last_leg_actions,
                foot_contacts,
                robot.projected_gravity,
                ee_goal_local_cart,
                locomotion_obs,
                np.array([manip_pred_last]),
                gait_indices.reshape(-1),
                clock_inputs.reshape(-1),
            ]
        )
        assert proprio.shape[0] == num_proprio, f"proprio dim {proprio.shape[0]} != {num_proprio}"

        if self.episode_length_buf <= 1:
            self.obs_history_buf = np.stack([proprio] * history_len, axis=0)
        else:
            self.obs_history_buf = np.concatenate(
                [self.obs_history_buf[1:], proprio[np.newaxis, :]], axis=0,
            )

        history_flat = self.obs_history_buf.reshape(-1)
        obs_vector = np.concatenate([proprio, self.privileged_obs, history_flat])

        slices = self._named_slices(proprio)
        return {
            "obs_vector": obs_vector.astype(np.float64),
            "proprio": proprio,
            "privileged": self.privileged_obs.copy(),
            "history": history_flat,
            "slices": slices,
            "ee_goal_local_cart": ee_goal_local_cart,
            "ee_goal_base": goal_base,
            "ee_goal_world": goal_base,  # back-compat alias (base frame, not sport world)
            "dvel_b_local": self.dvel_b_local.copy(),
            "commands": self.commands.copy(),
            "locomotion_obs": locomotion_obs.copy(),
            "use_cmd": self.use_cmd,
            "phase_variable": self.phase_variable,
            "manip_det": manip_det,
            "manip_det_pred": self.manip.manip_det_pred.copy(),
            "gait_indices": gait_indices,
            "clock_inputs": clock_inputs,
        }

    def _named_slices(self, proprio: np.ndarray) -> Dict[str, np.ndarray]:
        idx = 0
        locomotion_name = "commands" if self.use_cmd else "dvel_b_local"
        names = [
            ("body_rp", 2),
            ("base_lin_vel", 3),
            ("base_ang_vel", 3),
            ("dof_pos_err", 18),
            ("dof_vel", 18),
            ("last_leg_actions", 12),
            ("foot_contacts", 4),
            ("projected_gravity", 3),
            ("ee_goal_local_cart", 3),
            (locomotion_name, 3),
            ("manip_det_pred", 1),
            ("gait_indices", 1),
            ("clock_inputs", 4),
        ]
        out = {}
        for name, size in names:
            out[name] = proprio[idx : idx + size]
            idx += size
        return out
