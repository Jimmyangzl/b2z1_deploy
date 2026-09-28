"""Simplified dm_control task for b2z1 VR teleoperation."""

import collections

import mujoco
import numpy as np
import torch
from dm_control.suite import base

from constants import DT, XML_DIR
from controller_loader import (
    B2Z1WBCRoughCfg,
    WBCLoader,
    class_to_dict,
    device,
    history_len,
    num_proprio,
)
from helperfunc.helper_funcs import (
    orientation_error,
    quat_apply,
    quat_mul,
    quat_rotate_inverse_np,
    quaternion_to_rpy,
    rpy_to_quaternion,
)
from teleop.vr_pose_mapper import VrPoseMapper


class TeleopTask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 0
        self.vr_mapper = VrPoseMapper()
        self.wbc_policy = None

    def load_wbc(self, log_pth="wbc_ckpts/b2z1/lambdawbc-walk/model_100000.pt"):
        loader_cfg = B2Z1WBCRoughCfg()
        loader_cfg_dict = class_to_dict(loader_cfg)
        wbcloader = WBCLoader(train_cfg=loader_cfg_dict, device=device)
        wbcloader.load(log_pth)
        self.wbc_policy = wbcloader.get_inference_policy()

    def wbc_step(self, obs):
        wbc_action = self.wbc_policy(
            torch.tensor(obs["wbc"]).float().to(device),
            hist_encoding=True,
        )
        return wbc_action

    def update_vr_goal(self, vr_pose):
        base_yaw = quaternion_to_rpy(self.base_quat.copy())[2]
        goal_pos, goal_quat = self.vr_mapper.update(
            vr_pose,
            self.ee_pos.copy(),
            self.ee_quat.copy(),
            base_yaw,
        )
        self.curr_ee_goal_cart_world = goal_pos
        self.ee_goal_orn_quat = goal_quat
        self.ee_pos_goal_world = goal_pos.copy()
        self.ee_quat_goal_world = goal_quat.copy()
        (
            self.ee_pos_goal_local,
            self.ee_quat_goal_local,
        ) = self.world_to_base_transform(goal_pos, goal_quat)
        (
            self.ee_pos_goal_base_yaw,
            self.ee_quat_goal_base_yaw,
        ) = self.world_to_base_yaw_transform(goal_pos, goal_quat)

    def initialize_episode(self, physics):
        super().initialize_episode(physics)
        self._init_root_buffer(physics)
        self.vr_mapper.reset()

    def before_step(self, actions, physics):
        self.episode_length_buf += 1
        self.action_history_buf = np.vstack([self.action_history_buf[1:], actions])
        actions = self.action_history_buf[-1].copy()

        dpos = self.ee_pos_goal_world - self.ee_pos.copy()
        dpos_base_yaw = self.ee_pos_goal_base_yaw - self.world_to_base_yaw_transform(
            self.ee_pos.copy(), self.ee_quat.copy()
        )[0]
        dpos_xy_norm = np.linalg.norm(dpos_base_yaw[:2], ord=2)
        self.phase_variable = 1.0 / (1 + np.exp(-5 * (dpos_xy_norm - 1.0) / 1.0))

        dpos_norm = np.linalg.norm(dpos[:3], ord=2, keepdims=True)
        phantom_mask = dpos_norm > 0.2
        scale_factors = np.where(phantom_mask, 0.1 / dpos_norm, np.ones_like(dpos_norm))
        dpos_phantom = scale_factors * dpos

        self.dpos_b_local = self.ee_pos_goal_world - self.base_pos
        dvel_b_local_norm = np.linalg.norm(self.dpos_b_local[:2], ord=2, keepdims=True)
        self.dvel_b_local[:2] = self.dpos_b_local[:2] / (dvel_b_local_norm + 1e-8)
        mask = (self.manip_det_pred[-1] > 0.0001) & (self.phase_variable < 0.5)
        self.dvel_b_local[:2] = np.where(
            mask, np.zeros_like(self.dvel_b_local[:2]), self.dvel_b_local[:2]
        )
        if self.vr_mapper.mode == self.vr_mapper.MODE_IDLE:
            self.dvel_b_local[:] = 0.0

        drot = orientation_error(
            self.ee_quat_goal_world,
            self.ee_quat.copy() / np.linalg.norm(self.ee_quat),
        )
        dpose = np.append(dpos_phantom, drot)
        arm_pos_targets = self._control_ik(dpose) + self.qpos[12:18].copy()

        np.copyto(physics.named.data.ctrl[:12], self._compute_torques(actions[0:18]))
        np.copyto(
            physics.named.data.ctrl[12:19],
            np.append(arm_pos_targets, self.default_dof_pos[18]),
        )
        self.update_goal_frame(
            physics, "goal_frame", self.ee_pos_goal_world, self.ee_quat_goal_world
        )

    def get_observation(self, physics):
        self.foot_contacts_from_sensor = np.linalg.norm(self.foot_touch_force, axis=1) > 1.5
        self._step_contact_targets()

        base_yaw = quaternion_to_rpy(self.base_quat.copy())[2]
        self.base_yaw_quat = rpy_to_quaternion(np.array([0, 0, base_yaw]))
        arm_base_pos = self.base_pos + quat_apply(self.base_yaw_quat, self.arm_base_offset)
        ee_goal_local_cart = quat_rotate_inverse_np(
            self.base_quat, self.curr_ee_goal_cart_world - arm_base_pos
        )

        mujoco.mj_jacSite(
            physics.model.ptr,
            physics.data.ptr,
            self.ee_full_jac[:3],
            self.ee_full_jac[3:],
            self.ee_site_id,
        )
        self.ee_masked_jac = self.ee_full_jac[:, self.jac_dof_mask]
        self._compute_manipulability()
        self.manip_det_hist = np.append(self.manip_det_hist[1:], self.manip_det)
        self._predict_manipulability()

        self.base_lin_vel_local = quat_rotate_inverse_np(self.base_quat, self.base_lin_vel)
        self.base_ang_vel_local = quat_rotate_inverse_np(self.base_quat, self.base_ang_vel)
        self.projected_gravity = quat_rotate_inverse_np(self.base_quat, self.gravity_vec)

        obs = collections.OrderedDict()
        obs_buffer = np.concatenate(
            (
                self.get_body_rp(),
                self.base_lin_vel_local,
                self.base_ang_vel_local,
                (self.qpos.copy() - self.default_dof_pos)[:-1],
                self.qvel.copy()[:-1] * 0.05,
                self.action_history_buf[-1][:12],
                self.foot_contacts_from_sensor,
                self.projected_gravity,
                ee_goal_local_cart,
                self.dvel_b_local,
                self.manip_det_pred[-1:],
                self.gait_indices,
                self.clock_inputs,
            )
        )
        if self.episode_length_buf <= 1:
            self.obs_history_buf = np.stack([obs_buffer] * history_len, axis=0)
        else:
            self.obs_history_buf = np.concatenate(
                [self.obs_history_buf[1:], obs_buffer[np.newaxis, :]], axis=0
            )
        obs_buffer = np.concatenate(
            [obs_buffer, self.obs_history_buf.reshape(-1)], axis=-1
        )
        obs["wbc"] = np.expand_dims(obs_buffer, axis=0)
        obs["images"] = {
            "wrist": physics.render(height=480, width=640, camera_id="wrist_cam"),
        }
        ee_pos, ee_quat = self.world_to_base_transform(
            self.ee_pos.copy(), self.ee_quat.copy()
        )
        obs["ee_pose"] = np.append(ee_pos, ee_quat)
        obs["ee_world"] = np.append(self.ee_pos.copy(), self.ee_quat.copy())
        obs["ee_pose_d"] = np.append(self.ee_pos_goal_world, self.ee_quat_goal_world)
        obs["q_gripper"] = self.qpos[-1].copy()
        obs["q_gripper_d"] = self.default_dof_pos[18]
        return obs

    def get_reward(self, physics):
        return 0

    def get_body_rp(self):
        return quaternion_to_rpy(self.base_quat.copy())[:2]

    def _init_root_buffer(self, physics):
        self.base_pos = physics.named.data.xpos["base_link"]
        self.base_quat = physics.named.data.xquat["base_link"]
        self.base_lin_vel = physics.named.data.cvel["base_link"][3:6]
        self.base_ang_vel = physics.named.data.cvel["base_link"][:3]
        self.ee_pos = physics.named.data.xpos["ee_gripper_link"]
        self.ee_quat = physics.named.data.xquat["ee_gripper_link"]
        self.qpos = physics.named.data.qpos[7:26]
        self.qvel = physics.named.data.qvel[6:25]

        np.copyto(physics.named.data.qpos[0:2], np.array([0.1, 0.0]))
        physics.forward()

        num_sensors = physics.model.nsensor
        sensor_names = [physics.model.sensor(i).name for i in range(num_sensors)]
        touch_sensors = [name for name in sensor_names if "force" in name]
        self.foot_touch_force = np.array(
            [physics.named.data.sensordata[sensor_name] for sensor_name in touch_sensors]
        )
        self.gravity_vec = np.array([0.0, 0.0, -1.0])
        self.foot_contacts_from_sensor = np.linalg.norm(self.foot_touch_force, axis=1) > 1.5
        self.action_history_buf = np.zeros((3 + 2, 12 + 6 + 2))
        self.projected_gravity = quat_rotate_inverse_np(self.base_quat, self.gravity_vec)

        self.ee_site_id = physics.model.site("end_effector").id
        self.ee_full_jac = np.zeros((6, physics.model.nv))
        mujoco.mj_jacSite(
            physics.model.ptr,
            physics.data.ptr,
            self.ee_full_jac[:3],
            self.ee_full_jac[3:],
            self.ee_site_id,
        )
        z1_joint_name = "z1_joint1"
        z1_joint_id = physics.model.name2id(z1_joint_name, "joint")
        z1_joint_dofadr = physics.model.jnt_dofadr[z1_joint_id]
        self.jac_dof_mask = np.zeros(physics.model.nv, dtype=bool)
        self.jac_dof_mask[z1_joint_dofadr : z1_joint_dofadr + 6] = True
        self.ee_masked_jac = self.ee_full_jac[:, self.jac_dof_mask]

        self.manip_det = 0.0
        self.manip_det_pred = np.zeros(5)
        self.manip_det_hist = 0.0005 * np.ones(20)
        self.ee_goal_center_offset = np.array([0.2, 0.0, 0.55])
        self.dt = DT
        self.arm_base_offset = np.array([0.3, 0.0, 0.09])
        self.phase_variable = 0.0
        self.dvel_b_local = np.zeros(3)
        self.gait_indices = np.array([0.0])
        self.clock_inputs = np.array([0.0, 0.0, 0.0, 0.0])

        self.default_joint_angles = {
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FL_calf_joint": -1.5,
            "FR_hip_joint": 0.0,
            "FR_thigh_joint": 0.8,
            "FR_calf_joint": -1.5,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.8,
            "RL_calf_joint": -1.5,
            "RR_hip_joint": 0.0,
            "RR_thigh_joint": 0.8,
            "RR_calf_joint": -1.5,
            "z1_waist": 0.0,
            "z1_shoulder": 0.8 * 2 * np.pi / 3,
            "z1_elbow": -np.pi / 3,
            "z1_wrist_angle": 0.0,
            "z1_forearm_roll": 0.0,
            "z1_wrist_rotate": 1.57,
            "z1_jointGripper": -1.57018976,
        }
        self.default_dof_pos = np.zeros(len(self.default_joint_angles))
        for i, (_, value) in enumerate(self.default_joint_angles.items()):
            self.default_dof_pos[i] = value
        physics.named.data.qpos[7:26] = self.default_dof_pos

        self.obs_history_buf = np.zeros((history_len, num_proprio))
        self.episode_length_buf = -1
        self.action_scale = [0.4, 0.45, 0.45] * 4 + [2.1, 0.6, 0.6, 0, 0, 0]
        self.p_gains = 80
        self.d_gains = 2

        self.curr_ee_goal_cart_world = self.ee_pos.copy()
        self.ee_goal_orn_quat = self.ee_quat.copy()
        self.ee_pos_goal_world = self.ee_pos.copy()
        self.ee_quat_goal_world = self.ee_quat.copy()
        (
            self.ee_pos_goal_local,
            self.ee_quat_goal_local,
        ) = self.world_to_base_transform(self.ee_pos.copy(), self.ee_quat.copy())
        (
            self.ee_pos_goal_base_yaw,
            self.ee_quat_goal_base_yaw,
        ) = self.world_to_base_yaw_transform(self.ee_pos.copy(), self.ee_quat.copy())

    def world_to_base_transform(self, pos_world, quat_world):
        base_quat = self.base_quat.copy()
        base_quat_inv = base_quat.copy()
        base_quat_inv[1:] = -base_quat_inv[1:]
        world_pos_relative = pos_world - self.base_pos.copy()
        base_pos_goal = quat_rotate_inverse_np(base_quat, world_pos_relative)
        base_quat_goal = quat_mul(base_quat_inv, quat_world)
        return base_pos_goal, base_quat_goal

    def world_to_base_yaw_transform(self, pos_world, quat_world):
        base_rpy = quaternion_to_rpy(self.base_quat.copy())
        base_yaw_quat = rpy_to_quaternion(np.array([0.0, 0.0, base_rpy[2]]))
        base_yaw_quat_inv = base_yaw_quat.copy()
        base_yaw_quat_inv[1:] = -base_yaw_quat_inv[1:]
        world_pos_relative = pos_world - self.base_pos.copy()
        base_yaw_pos_goal = quat_rotate_inverse_np(base_yaw_quat, world_pos_relative)
        base_yaw_quat_goal = quat_mul(base_yaw_quat_inv, quat_world)
        return base_yaw_pos_goal, base_yaw_quat_goal

    def _control_ik(self, dpose):
        j_eef_T = self.ee_masked_jac.T
        lmbda = np.eye(6) * (0.05 ** 2)
        A = self.ee_masked_jac @ j_eef_T + lmbda
        u = j_eef_T @ np.linalg.solve(A, dpose)
        return u.flatten()

    def _compute_torques(self, actions):
        actions_scaled = actions * self.action_scale
        default_torques = (
            self.p_gains * (actions_scaled + (self.default_dof_pos - self.qpos.copy())[:-1])
            - self.d_gains * self.qvel.copy()[:-1]
        )
        return default_torques[:12]

    def _compute_manipulability(self):
        A = self.ee_masked_jac @ self.ee_masked_jac.T
        self.manip_det = np.linalg.det(A)

    @staticmethod
    def double_exponential_smoothing(x, alpha=0.5):
        seq_len = len(x)
        level = np.zeros(seq_len)
        trend_component = np.zeros(seq_len)
        level[0] = x[0]
        if seq_len > 1:
            trend_component[0] = x[1] - x[0]
        for t in range(1, seq_len):
            level[t] = alpha * x[t] + (1 - alpha) * (level[t - 1] + trend_component[t - 1])
            trend_component[t] = alpha * (level[t] - level[t - 1]) + (1 - alpha) * trend_component[t - 1]
        return level, trend_component

    def _predict_manipulability(self, forecast_steps=5):
        level, trend = self.double_exponential_smoothing(self.manip_det_hist)
        self.manip_det_pred = np.array(
            [level[-1] + (i + 1) * trend[-1] for i in range(forecast_steps)]
        )

    def _step_contact_targets(self):
        frequencies = 2
        phases = 0.5
        offsets = 0
        bounds = 0
        durations = 0.5
        dt = DT
        self.gait_indices = np.remainder(self.gait_indices + dt * frequencies, 1.0)
        self.gait_indices[~self._get_walking_cmd_mask()] = 0
        foot_indices = [
            self.gait_indices + phases + offsets + bounds,
            self.gait_indices + offsets,
            self.gait_indices + bounds,
            self.gait_indices + phases,
        ]
        self.foot_indices = np.remainder(np.column_stack(foot_indices), 1.0)
        for idxs in foot_indices:
            stance_idxs = np.remainder(idxs, 1) < durations
            swing_idxs = np.remainder(idxs, 1) > durations
            idxs[stance_idxs] = np.remainder(idxs[stance_idxs], 1) * (0.5 / durations)
            idxs[swing_idxs] = 0.5 + (np.remainder(idxs[swing_idxs], 1) - durations) * (
                0.5 / (1 - durations)
            )
        self.clock_inputs[0] = np.sin(2 * np.pi * foot_indices[0])
        self.clock_inputs[1] = np.sin(2 * np.pi * foot_indices[1])
        self.clock_inputs[2] = np.sin(2 * np.pi * foot_indices[2])
        self.clock_inputs[3] = np.sin(2 * np.pi * foot_indices[3])

    def _get_walking_cmd_mask(self):
        walking_mask0 = np.abs(self.dvel_b_local[0]) > 0.02
        walking_mask1 = np.abs(self.dvel_b_local[1]) > 0.02
        walking_mask2 = np.abs(self.dvel_b_local[2]) > 0.05
        return walking_mask0 | walking_mask1 | walking_mask2

    def update_goal_frame(self, physics, body_name, position, quaternion):
        np.copyto(physics.data.mocap_pos[0], np.array(position.copy()))
        np.copyto(physics.data.mocap_quat[0], np.array(quaternion.copy()))
