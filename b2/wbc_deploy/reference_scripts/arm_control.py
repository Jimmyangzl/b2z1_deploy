import numpy as np
import torch
from isaacgym.torch_utils import *


class ArmController:
    """Arm IK, RFM phase logic, and locomotion command generation for lambdaWBC."""

    ARM_JOINT_NAMES = (
        "z1_waist",
        "z1_shoulder",
        "z1_elbow",
        "z1_wrist_angle",
        "z1_forearm_roll",
        "z1_wrist_rotate",
    )

    def __init__(self, env):
        self.env = env
        defaults = env.cfg.init_state.default_joint_angles
        self.arm_home_joint = torch.tensor(
            [float(defaults[name]) for name in self.ARM_JOINT_NAMES],
            device=env.device,
            dtype=torch.float,
        )

    @property
    def device(self):
        return self.env.device

    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def cfg(self):
        return self.env.cfg

    def init_buffers(self):
        self.dpos_b_world = torch.zeros(self.num_envs, 2, device=self.device)
        self.dpos_b_local = torch.zeros(self.num_envs, 2, device=self.device)
        self.dvel_b_world = torch.zeros(self.num_envs, 3, device=self.device)
        self.dvel_b_local = torch.zeros(self.num_envs, 3, device=self.device)
        self.lambda_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self.phase_variable = torch.zeros(self.num_envs, 1, device=self.device)

    def process_actions(self, actions):
        env = self.env
        actions = actions.clone()
        if env.num_actions <= 12:
            # Legs-only policy; arm pose via IK.
            actions = torch.clip(actions[:, :12], -env.clip_actions, env.clip_actions).to(self.device)
        else:
            # Full policy (legs + arm / lambda). With use_cmd, zero non-leg outputs.
            if self.cfg.experiment.use_cmd:
                actions[:, 12:env.num_actions] = 0.0
            else:
                actions[:, 12:18] = 0.0
            actions[:, :18] = torch.clip(actions[:, :18], -env.clip_actions, env.clip_actions).to(self.device)
            if env.num_actions >= 20:
                actions[:, 18:20] = torch.clip(actions[:, 18:20], 0.0, 100.0).to(self.device)
            actions = actions[:, :env.num_actions]

        if env.action_delay != -1:
            env.action_history_buf = torch.cat(
                [env.action_history_buf[:, 1:], actions[:, None, :]], dim=1,
            )
        if env.global_steps < 10000 * 24:
            actions = env.action_history_buf[:, -1]
        else:
            actions = env.action_history_buf[:, -2]
        return actions

    def update_rfm_state(self, actions):
        env = self.env
        if actions.shape[-1] >= 20:
            self.lambda_xy = (actions[:, 18:20] - 50) / 100.0
        # else: keep lambda_xy at zeros (legs-only policy)
        dpos = env.curr_ee_goal_cart_world - env.ee_pos
        if self.cfg.experiment.use_rfm:
            dpos_xy_norm = torch.norm(dpos[:, :2], p=2, dim=1)
            self.phase_variable = 1.0 / (1 + torch.exp(-5 * (dpos_xy_norm - 1.0) / 1.0))

        # dvel is used for RFM locomotion obs/rewards when use_cmd=False.
        # Previously gated on fix_sample only, which left dvel=0 under base_sample.
        need_dvel = (not self.cfg.experiment.use_cmd) and (
            self.cfg.experiment.fix_sample
            or bool(getattr(self.cfg.experiment, "base_sample", False))
        )
        if not need_dvel:
            return

        dpos_b_world = env.curr_ee_goal_cart_world.clone() - env.base_pos.clone()
        self.dpos_b_local = quat_rotate_inverse(env.base_yaw_quat, dpos_b_world)

        dvel_b_local_norm = torch.norm(self.dpos_b_local[:, :2], p=2, dim=1, keepdim=True)
        self.dvel_b_local[:, :2] = 0.2 * self.dpos_b_local[:, :2] / (dvel_b_local_norm[:, :2] + 1e-8)
        self.dvel_b_local[:, 2] = 0.4 * env.manip.yaw_direction[:, 0]

        _, _, ee_goal_yaw = euler_from_quat(env.ee_goal_orn_quat)
        yaw_diff = torch.atan2(
            torch.sin(ee_goal_yaw - env.base_yaw_euler[:, 2]),
            torch.cos(ee_goal_yaw - env.base_yaw_euler[:, 2]),
        )
        ee_euler_error = torch.abs(yaw_diff)
        # self.phase_variable = 0.3
        mask_xy = (
            (env.manip.manip_det_pred[:, -1] > 0.0003) & (self.phase_variable < 0.5)
        ).unsqueeze(-1).expand_as(self.dvel_b_local[:, :2])
        mask_yaw = (
            # (env.manip.manip_det_pred[:, -1] < 0.001)
            # & (self.phase_variable < 0.5)
            # & (ee_euler_error > 0.1)
            (ee_euler_error > 0.785)
        )
        # print(mask_yaw)

        if bool(getattr(self.cfg.experiment, "base_sample", False)):
            arm_base_pos = env.base_pos + quat_apply(env.base_quat, env.arm_base_offset)
        else:
            arm_base_pos = env.base_pos + quat_apply(env.base_yaw_quat, env.arm_base_offset)
        dist_base_ee = torch.norm(env.ee_pos[:, :2] - arm_base_pos[:, :2], p=2, dim=1)

        mask_back = (
            (env.manip.manip_det_pred[:, -1] < 0.0003)
            & (dist_base_ee < 0.4)
            & (self.phase_variable < 0.5)
        ).unsqueeze(-1).expand_as(self.dvel_b_local[:, :2])


        self.dvel_b_local[:, :2] = torch.where(mask_xy, torch.zeros_like(self.dvel_b_local[:, :2]), self.dvel_b_local[:, :2])
        self.dvel_b_local[:, :2] = torch.where(mask_back, -self.dvel_b_local[:, :2], self.dvel_b_local[:, :2])
        self.dvel_b_local[:, 2] = torch.where(mask_yaw, self.dvel_b_local[:, 2], torch.zeros_like(self.dvel_b_local[:, 2]))

    def control_ik(self, dpose):
        env = self.env
        j_eef_T = torch.transpose(env.ee_j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (0.05 ** 2)
        A = torch.bmm(env.ee_j_eef, j_eef_T) + lmbda[None, ...]
        u = torch.bmm(j_eef_T, torch.linalg.solve(A, dpose))
        return u.squeeze(-1)

    def compute_pos_targets(self, actions):
        env = self.env
        dpos = env.curr_ee_goal_cart_world - env.ee_pos
        dpos_norm = torch.norm(dpos[:, :3], p=2, dim=1, keepdim=True)
        phantom_mask = dpos_norm > 0.1
        scale_factors = torch.where(phantom_mask, 0.1 / dpos_norm, torch.ones_like(dpos_norm))
        dpos_phantom = scale_factors * dpos

        drot = orientation_error(
            env.ee_goal_orn_quat,
            env.ee_orn / torch.norm(env.ee_orn, dim=-1).unsqueeze(-1),
        )
        dpose = torch.cat([dpos_phantom, drot], -1).unsqueeze(-1)

        gripper_joints = self.cfg.env.num_gripper_joints
        arm_slice = slice(-(6 + gripper_joints), -gripper_joints)
        gripper_slice = slice(-gripper_joints, None) if gripper_joints > 0 else None

        # Default: IK toward EE goal (walk / unfixed arms).
        arm_pos_targets = self.control_ik(dpose) + env.dof_pos[:, arm_slice]
        if not self.cfg.experiment.use_cmd and self.cfg.experiment.flag_inference:
            home_arm_mask = self.phase_variable > 0.5
            arm_pos_targets[home_arm_mask, :] = self.arm_home_joint

        # fix_arm: freeze only table/wall envs at default joints; walk keeps IK.
        if self.cfg.experiment.fix_arm and hasattr(env, "env_mode"):
            climb_mask = env.env_mode != env.MODE_WALK
            if climb_mask.any():
                arm_pos_targets[climb_mask] = env.default_dof_pos[arm_slice]

        # z1_elbow (arm index 2): keep above URDF lower limit.
        arm_pos_targets[:, 2] = torch.clamp(arm_pos_targets[:, 2], min=-3.0)

        all_pos_targets = torch.zeros_like(env.dof_pos)
        all_pos_targets[:, arm_slice] = arm_pos_targets
        if gripper_slice is not None:
            all_pos_targets[:, gripper_slice] = env.default_dof_pos[gripper_slice]
        return all_pos_targets
