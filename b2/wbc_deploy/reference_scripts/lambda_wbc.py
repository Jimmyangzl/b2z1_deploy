# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym_wbc.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float

import torch
from typing import Tuple, Dict

from legged_gym_wbc.envs.base.legged_robot import LeggedRobot
from legged_gym_wbc import LEGGED_GYM_ROOT_DIR
from legged_gym_wbc.utils.helpers import class_to_dict
from legged_gym_wbc.utils.terrain import Terrain, Terrain_Perlin
from .b2z1_lambdawbc_config import B2Z1WBCRoughCfg
from .ee_goal_manager import EeGoalManager
from .manipulability import ManipulabilityTracker
from .arm_control import ArmController

import sys
import xml.etree.ElementTree as ET
from urdf_parser_py.urdf import URDF

import csv

_GOAL_DELEGATE_ATTRS = frozenset({
    'traj_timesteps', 'traj_total_timesteps', 'goal_timer',
    'ee_start_sphere', 'ee_goal_sphere', 'ee_start_cart', 'ee_goal_cart',
    'centers_fixed', 'base_yaw_quat_fixed', 'ee_yaw_quat_fixed',
    'ee_goal_orn_euler', 'ee_goal_orn_quat', 'ee_goal_orn_delta_rpy',
    'curr_ee_goal_cart', 'curr_ee_goal_sphere', 'curr_ee_goal_cart_world',
    'init_start_ee_sphere', 'init_end_ee_sphere',
    'sample_category',
    'collision_lower_limits', 'collision_upper_limits', 'underground_limit',
    'num_collision_check_samples', 'collision_check_t',
    'sphere_error_scale', 'orn_error_scale', 'ee_goal_center_offset',
    'reset_tele_op_flag',
})

_MANIP_DELEGATE_ATTRS = frozenset({
    'manip_det', 'manip_det_pred', 'manip_det_hist',
    'manip_xy', 'manip_xy_pred', 'manip_xy_hist',
    'manip_yaw', 'manip_yaw_pred', 'manip_yaw_hist',
    'yaw_direction',
})

_ARM_DELEGATE_ATTRS = frozenset({
    'dpos_b_world', 'dpos_b_local', 'dvel_b_world', 'dvel_b_local',
    'lambda_xy', 'phase_variable', 'arm_home_joint',
})

class lambdaWBC(LeggedRobot):
    name = None
    # cfg: B1Z1RoughCfg
    # cfg: B2Z1WBCRoughCfg

    def __init__(self, cfg, *args, **kwargs):
        if cfg.env.observe_gait_commands:
            print("||||||||||Observe gait commands!")
            cfg.env.num_proprio += 5 # gait_indices=1, clock_phase=4
            cfg.env.num_observations = cfg.env.num_proprio * (cfg.env.history_len+1) + cfg.env.num_priv
            self.num_obs = cfg.env.num_observations
        self.stand_by = cfg.env.stand_by
        super().__init__(cfg, *args, **kwargs)

        self.sampleFlag = -torch.ones(self.num_envs, 1, device=self.device)
        self.updatelmbdaCount = 10
            
    def __getattr__(self, name):
        if name in _GOAL_DELEGATE_ATTRS:
            ee_goals = self.__dict__.get('ee_goals')
            if ee_goals is not None:
                return getattr(ee_goals, name)
        if name in _MANIP_DELEGATE_ATTRS:
            manip = self.__dict__.get('manip')
            if manip is not None:
                return getattr(manip, name)
        if name in _ARM_DELEGATE_ATTRS:
            arm = self.__dict__.get('arm')
            if arm is not None:
                return getattr(arm, name)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
            
    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        self.render()
        actions = self.arm.process_actions(actions)
        self.actions = actions.clone()
        self.arm.update_rfm_state(actions)
        all_pos_targets = self.arm.compute_pos_targets(actions)
        all_pos_targets[:, 12:18] = self.default_dof_pos_wo_gripper[12:18]
        # Leg PD uses an 18-wide torque vector (arm slots zeroed when unused).
        torques_actions = torch.zeros(self.num_envs, self.num_torques, device=self.device)
        n = min(self.actions.shape[-1], self.num_torques)
        torques_actions[:, :n] = self.actions[:, :n]
        for t in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(torques_actions)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(all_pos_targets))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_jacobian_tensors(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        self.global_steps += 1
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.arm_rew_buf, self.reset_buf, self.extras

    def tele_ref(self, ref_pos, ref_orn, reset_flag):
        self.ref_pos = ref_pos
        self.ref_orn = ref_orn
        self.ee_goals.reset_tele_op_flag = reset_flag

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.episode_length_buf += 1
        self.common_step_counter += 1
        
        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        base_yaw = euler_from_quat(self.base_quat)[2]
        self.base_yaw_euler[:] = torch.cat([torch.zeros(self.num_envs, 2, device=self.device), base_yaw.view(-1, 1)], dim=1)
        self.base_yaw_quat[:] = quat_from_euler_xyz(torch.tensor(0), torch.tensor(0), base_yaw)
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 2.
        self.contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact

        self.foot_contacts_from_sensor = self.force_sensor_tensor.norm(dim=-1) > 10.   
        
        self._post_physics_step_callback()

        if self.cfg.debug.test_control:
            self.log_errors(0)
        
        # update ee goal
        if not self.cfg.experiment.fix_arm:
            self.ee_goals.update_curr()

        
        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if self.reset_tele_op_flag:
            env_ids = 0
        self.reset_idx(env_ids, start=False)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)
        
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_torques[:] = self.torques[:]

        if (self.viewer and self.enable_viewer_sync and self.debug_viz) or self.record_video:
            self.gym.clear_lines(self.viewer)
            self._draw_ee_goal_curr()

    def _accumulate_rewards(self, reward_buf, functions, names, scales, post_scale=None):
        prio_state = {}
        for i in range(len(functions)):
            name = names[i]
            rew, metric = functions[i]()
            rew = rew * scales[name]
            if post_scale is not None:
                rew, metric = post_scale(name, rew, metric, prio_state)
            reward_buf += rew
            self.episode_sums[name] += rew
            self.episode_metric_sums[name] += metric

    def _loco_reward_post_scale(self, name, rew, metric, prio_state):
        if not self.cfg.experiment.use_prio_reward:
            return rew, metric
        if name == 'loco_base_height':
            prio_state['track'] = (rew, metric)
        elif name in ('tracking_v_base_max', 'tracking_ang_v_base'):
            if 'track' in prio_state:
                track_rew, track_metric = prio_state['track']
                rew = track_rew * rew
                metric = track_metric * metric
        rew = self.phase_variable * rew
        return rew, metric

    def _arm_reward_post_scale(self, name, rew, metric, prio_state):
        if not self.cfg.experiment.use_prio_reward:
            return rew, metric
        if name == 'tracking_dvel_b_local_max':
            prio_state['track'] = (rew, metric)
        elif name == 'tracking_ee_world':
            if 'track' in prio_state:
                track_rew, track_metric = prio_state['track']
                rew = track_rew * rew
                metric = track_metric * metric
        rew = (1 - self.phase_variable) * rew
        return rew, metric

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        self._accumulate_rewards(
            self.rew_buf, self.reward_functions, self.reward_names, self.reward_scales,
        )
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew, metric = self._reward_termination()
            rew = rew * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
            self.episode_metric_sums["termination"] += metric

        if self.cfg.experiment.use_rfm:
            self._accumulate_rewards(
                self.rew_buf, self.loco_reward_functions, self.loco_reward_names,
                self.loco_reward_scales, post_scale=self._loco_reward_post_scale,
            )

        self.rew_buf /= 100

        self.arm_rew_buf[:] = 0.
        self._accumulate_rewards(
            self.arm_rew_buf, self.arm_reward_functions, self.arm_reward_names,
            self.arm_reward_scales, post_scale=self._arm_reward_post_scale,
        )
        if self.cfg.rewards.only_positive_rewards:
            self.arm_rew_buf[:] = torch.clip(self.arm_rew_buf[:], min=0.)
        # add termination reward after clipping
        if "arm_termination" in self.arm_reward_scales:
            rew, metric = self._reward_termination()
            rew = rew * self.arm_reward_scales["arm_termination"]
            self.arm_rew_buf += rew
            self.episode_sums["arm_termination"] += rew
            self.episode_metric_sums["arm_termination"] += metric

        self.arm_rew_buf /= 100

    def _build_proprio_obs(self, ee_goal_local_cart):
        grip = self.cfg.env.num_gripper_joints
        # ee_goal_local_cart = torch.tensor([0.6, 0., 0.48], device=self.device).repeat(self.num_envs, 1)
        
        common = [
            self._get_body_orientation(),
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            ((self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos)[:, :-grip],
            (self.dof_vel * self.obs_scales.dof_vel)[:, :-grip],
            (self.action_history_buf[:, -1])[:, :12],
            self.foot_contacts_from_sensor,
            self.projected_gravity,
            ee_goal_local_cart,
        ]
        print("base_lin_vel[:, 0]")
        print(self.base_lin_vel[:, 0])
        print("pitch")
        pitch = self._get_body_orientation()[:, 1]
        print(pitch)
        if self.cfg.experiment.use_cmd:
            tail = [self.commands[:, :3] * self.commands_scale, self.manip_det_pred[:, -1].unsqueeze(-1)]
            # self.commands[:, :3] = 0 * self.commands[:, :3]
        else:
            tail = [self.dvel_b_local, self.manip_det_pred[:, -1].unsqueeze(-1)]
        # print(torch.cat(common + tail, dim=-1)[0])
        return torch.cat(common + tail, dim=-1)

    def compute_observations(self):
        """ Computes observations """
        arm_base_pos = self.base_pos + quat_apply(self.base_yaw_quat, self.arm_base_offset)
        ee_goal_local_cart = quat_rotate_inverse(self.base_quat, self.curr_ee_goal_cart_world - arm_base_pos)
        ee_goal_local_cart = quat_rotate_inverse(self.base_quat, self.ee_pos - arm_base_pos)
        self.manip.compute()
        self.manip.update_history()
        self.manip.predict()

        if self.stand_by:
            self.commands[:] = 0.
        
        obs_buf = self._build_proprio_obs(ee_goal_local_cart)
        if self.cfg.env.observe_gait_commands:
            obs_buf = torch.cat((obs_buf,
                self.gait_indices.unsqueeze(1), self.clock_inputs), dim=-1)
        
        if self.cfg.domain_rand.observe_priv:
            priv_buf = torch.cat((
                self.mass_params_tensor,
                self.friction_coeffs_tensor,
                self.motor_strength[:, :12] - 1,
            ), dim=-1)
            self.obs_buf = torch.cat([obs_buf, priv_buf, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        else:
            self.obs_buf = torch.cat([obs_buf, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        

        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
            
            torch.cat([
                self.obs_history_buf[:, 1:],
                obs_buf.unsqueeze(1)
            ], dim=1)
        )  

        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec


    def check_termination(self):
        """ Check if environments need to be reset
        """
        termination_contact_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)

        r, p, _ = euler_from_quat(self.base_quat) 
        z = self.root_states[:, 2]

        # r_threshold_buff = ((r > 0.2) & (self.curr_ee_goal_sphere[:, 2] >= 0)) | ((r < -0.2) & (self.curr_ee_goal_sphere[:, 2] <= 0))
        # p_threshold_buff = ((p > 0.2) & (self.curr_ee_goal_sphere[:, 1] >= 0)) | ((p < -0.2) & (self.curr_ee_goal_sphere[:, 1] <= 0))
        r_term = torch.abs(r) > 0.8
        p_term = torch.abs(p) > 0.8
        z_term = z < 0.1
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        # arm_base_local = torch.tensor([0.3, 0.0, 0.09], device=self.device).repeat(self.num_envs, 1)
        # arm_base = quat_apply(self.base_quat, arm_base_local) + self.root_states[:, :3]
        # curr_ee_pos_local = quat_rotate_inverse(self.root_states[:, 3:7], self.ee_pos - arm_base)
        # ik_fail = (self.curr_ee_goal_cart[:, -1:] - curr_ee_pos_local[:, -1:]).norm(dim=-1) > 0.3
        # self.reset_buf = termination_contact_buf | self.time_out_buf | r_term | p_term | z_term | ik_fail
        self.reset_buf = termination_contact_buf | self.time_out_buf | r_term | p_term | z_term

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self.terrain = Terrain(self.cfg.terrain, )
        self._create_trimesh()
        # self._create_ground_plane()
        self._create_envs()
        
        
    def reset_idx(self, env_ids, start=False):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        if self.cfg.debug.test_control and env_ids == 0:
            self.save_csv_final()
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        
        # reset robot states
        self._reset_root_states(env_ids)
        self._reset_dofs(env_ids)
        

        if start:
            command_env_ids = env_ids
        else:
            command_env_ids = self.time_out_buf.nonzero(as_tuple=False).flatten()
        self._resample_commands(command_env_ids)
        self.ee_goals.resample_on_reset(env_ids, is_init=True)
        
        # self._resample_ee_goal_fixed(command_env_ids)

        # reset buffers
        self.last_torques[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.obs_history_buf[env_ids, :, :] = 0.
        self.action_history_buf[env_ids, :, :] = 0.
        self.ee_goals.reset_goal_timer(env_ids)

        self.manip.reset(env_ids)


        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        
        for key in self.episode_metric_sums.keys():
            self.extras["episode"]['metric_' + key] = torch.mean(self.episode_metric_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_metric_sums[key][env_ids] = 0.

        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        
    # ------------ callbacks ------------

    def _parse_cfg(self, cfg):
        self.num_torques = self.cfg.env.num_torques
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.arm_reward_scales = class_to_dict(self.cfg.rewards.arm_scales)
        self.loco_reward_scales = class_to_dict(self.cfg.rewards.loco_scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        self.goal_ee_ranges = class_to_dict(self.cfg.goal_ee.ranges)

        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        self.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        self.clip_actions = self.cfg.normalization.clip_actions
        self.action_delay = self.cfg.env.action_delay
        self.stop_update_goal = self.cfg.env.stop_update_goal
        self.record_video = self.cfg.env.record_video
    
    def _load_joint_limits_from_urdf(self):
        urdf_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        robot = URDF.from_xml_file(urdf_path)
        joint_upper_limits = []
        joint_lower_limits = []
        
        for joint in robot.joints:
            if joint.limit is not None:
                joint_lower_limits.append(joint.limit.lower) 
                joint_upper_limits.append(joint.limit.upper)
        return joint_lower_limits, joint_upper_limits

    def _load_arm_base_offset_from_urdf(self):
        """Read arm mount offset from fixed joint base_static_joint in the URDF.

        Uses ElementTree instead of urdf_parser_py because the URDF contains
        arithmetic expressions (e.g. '0.47+0.66') that the parser cannot float().
        """
        urdf_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        root = ET.parse(urdf_path).getroot()
        for joint in root.findall("joint"):
            if joint.get("name") == "base_static_joint":
                origin = joint.find("origin")
                if origin is None or origin.get("xyz") is None:
                    raise ValueError(f"base_static_joint has no origin xyz in URDF: {urdf_path}")
                return [float(x) for x in origin.get("xyz").split()]
        raise ValueError(f"base_static_joint not found in URDF: {urdf_path}")

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        if self.cfg.debug.vl_train:
            from legged_gym_wbc.envs.rewards.maniploco_vl_rewards import ManipLoco_rewards
        else:
            from legged_gym_wbc.envs.rewards.maniploco_rewards import ManipLoco_rewards 

        reward_contrainers = {"maniploco_rewards": ManipLoco_rewards}
        self.reward_container = reward_contrainers[self.cfg.rewards.reward_container_name](self)
        self.reward_scales = {k:v for k,v in self.reward_scales.items() if v is not None and v != 0}

        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self.reward_container, name))

        self.arm_reward_scales = {k:v for k,v in self.arm_reward_scales.items() if v is not None and v != 0}
        # prepare list of functions
        self.arm_reward_functions = []
        self.arm_reward_names = []
        for name, scale in self.arm_reward_scales.items():
            if name=="termination":
                continue
            self.arm_reward_names.append(name)
            name = '_reward_' + name
            self.arm_reward_functions.append(getattr(self.reward_container, name))
        
        self.loco_reward_scales = {k:v for k,v in self.loco_reward_scales.items() if v is not None and v != 0}
        # prepare list of functions
        self.loco_reward_functions = []
        self.loco_reward_names = []
        for name, scale in self.loco_reward_scales.items():
            if name=="termination":
                continue
            self.loco_reward_names.append(name)
            name = '_reward_' + name
            self.loco_reward_functions.append(getattr(self.reward_container, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in list(self.reward_scales.keys()) + list(self.loco_reward_scales.keys()) + list(self.arm_reward_scales.keys())}

        self.episode_metric_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                        for name in list(self.reward_scales.keys()) + list(self.loco_reward_scales.keys()) + list(self.arm_reward_scales.keys())}

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        self.custom_origins = True
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # put robots at the origins defined by the terrain
        max_init_level = self.cfg.terrain.max_init_terrain_level  # start from 0
        if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
        self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
        self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
        self.max_terrain_level = self.cfg.terrain.num_rows
        self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
        self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
        print("Added ground plane with friction: {}, restitution: {}".format(plane_params.static_friction, plane_params.restitution))
        return
    
    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
            Very slow when horizontal_scale is small
        """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        print("Adding trimesh to simulation...")
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)  
        print("Trimesh added")
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity
        asset_options.use_mesh_materials = True

        # Robot
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dofs = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        dof_props_asset['driveMode'][12:].fill(gymapi.DOF_MODE_POS)  # set arm to pos control
        dof_props_asset['stiffness'][12:].fill(400.0)
        dof_props_asset['damping'][12:].fill(40.0)
        # dof_props_asset['driveMode'][:].fill(gymapi.DOF_MODE_EFFORT)  # set all joints to pos control
        # dof_props_asset['stiffness'][:].fill(100.0)
        # dof_props_asset['damping'][:].fill(5.0)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)
        self.body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.body_names_to_idx = self.gym.get_asset_rigid_body_dict(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.dof_wo_gripper_names = self.dof_names[:-self.cfg.env.num_gripper_joints]
        self.dof_names_to_idx = self.gym.get_asset_dof_dict(robot_asset)
        # self.num_bodies = len(self.body_names)
        # self.num_dofs = len(self.dof_names)
        feet_names = [s for s in self.body_names if self.cfg.asset.foot_name in s]
        print(self.body_names)
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            body_names = [s for s in self.body_names if name in s]
            if len(body_names) == 0:
                raise Exception('No body found with name {}'.format(name))
            penalized_contact_names.extend(body_names)
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            body_names = [s for s in self.body_names if name in s]
            if len(body_names) == 0:
                raise Exception('No body found with name {}'.format(name))
            termination_contact_names.extend(body_names)

        self.sensor_indices = []
        for name in feet_names:
            foot_idx = self.body_names_to_idx[name]
            sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, -0.05))
            sensor_idx = self.gym.create_asset_force_sensor(robot_asset, foot_idx, sensor_pose)
            self.sensor_indices.append(sensor_idx)
        
        self.gripper_idx = self.body_names_to_idx[self.cfg.asset.gripper_name]



        print('------------------------------------------------------')
        print('num_actions: {}'.format(self.num_actions))
        print('num_torques: {}'.format(self.num_torques))
        print('num_dofs: {}'.format(self.num_dofs))
        print('num_bodies: {}'.format(self.num_bodies))
        print('penalized_contact_names: {}'.format(penalized_contact_names))
        print('termination_contact_names: {}'.format(termination_contact_names))
        print('feet_names: {}'.format(feet_names))
        print(f"EE Gripper index: {self.gripper_idx}")

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])
        # box_start_pose = gymapi.Transform()

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.mass_params_tensor = torch.zeros(self.num_envs, 5, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            self.envs.append(env_handle)

            # robot 
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-self.cfg.init_state.origin_perturb_range, self.cfg.init_state.origin_perturb_range, (2,1), device=self.device).squeeze(1)
            rand_yaw_quat = gymapi.Quat.from_euler_zyx(0., 0., self.cfg.init_state.rand_yaw_range*np.random.uniform(-1, 1))
            start_pose.r = rand_yaw_quat
            start_pose.p = gymapi.Vec3(*pos)
            
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            robot_dog_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, "robot_dog", i, self.cfg.asset.self_collisions, 0)
            self.actor_handles.append(robot_dog_handle)

            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, robot_dog_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, robot_dog_handle)
            body_props, mass_params = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, robot_dog_handle, body_props, recomputeInertia=True)
            
            self.mass_params_tensor[i, :] = torch.from_numpy(mass_params).to(self.device)

        assert(np.all(np.array(self.actor_handles) == 0))
        self.robot_actor_indices = torch.arange(0, 2 * self.num_envs, 2, device=self.device)

        self.friction_coeffs_tensor = self.friction_coeffs.to(self.device).squeeze(-1)

        if self.cfg.domain_rand.randomize_motor:
            self.motor_strength = torch.cat([
                    torch_rand_float(self.cfg.domain_rand.leg_motor_strength_range[0], self.cfg.domain_rand.leg_motor_strength_range[1], (self.num_envs, 12), device=self.device),
                    torch_rand_float(self.cfg.domain_rand.arm_motor_strength_range[0], self.cfg.domain_rand.arm_motor_strength_range[1], (self.num_envs, 6), device=self.device)
                ], dim=1)
        else:
            self.motor_strength = torch.ones(self.num_envs, self.num_torques, device=self.device)

        hip_names = ["FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint"]
        self.hip_indices = torch.zeros(len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(hip_names):
            self.hip_indices[i] = self.dof_names.index(name)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalized_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalized_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
        
        print('penalized_contact_indices: {}'.format(self.penalized_contact_indices))
        print('termination_contact_indices: {}'.format(self.termination_contact_indices))
        print('feet_indices: {}'.format(self.feet_indices))
        
        if self.record_video:
            camera_props = gymapi.CameraProperties()
            camera_props.width = 720
            camera_props.height = 480
            self._rendering_camera_handles = []
            for i in range(self.num_envs):
                # root_pos = self.root_states[i, :3].cpu().numpy()
                # cam_pos = root_pos + np.array([0, 1, 0.5])
                cam_pos = np.array([0, 1, 0.5])
                camera_handle = self.gym.create_camera_sensor(self.envs[i], camera_props)
                self._rendering_camera_handles.append(camera_handle)
                self.gym.set_camera_location(camera_handle, self.envs[i], gymapi.Vec3(*cam_pos), gymapi.Vec3(*0*cam_pos))
    
    def _process_rigid_body_props(self, props, env_id):
        if self.cfg.domain_rand.randomize_base_mass:
            rng_mass = self.cfg.domain_rand.added_mass_range
            rand_mass = np.random.uniform(rng_mass[0], rng_mass[1], size=(1, ))
            props[1].mass += rand_mass
        else:
            rand_mass = np.zeros(1)
        
        if self.cfg.domain_rand.randomize_gripper_mass:
            gripper_rng_mass = self.cfg.domain_rand.gripper_added_mass_range
            gripper_rand_mass = np.random.uniform(gripper_rng_mass[0], gripper_rng_mass[1], size=(1, ))
            props[self.gripper_idx].mass += gripper_rand_mass
        else:
            gripper_rand_mass = np.zeros(1)

        if self.cfg.domain_rand.randomize_base_com:
            rng_com_x = self.cfg.domain_rand.added_com_range_x
            rng_com_y = self.cfg.domain_rand.added_com_range_y
            rng_com_z = self.cfg.domain_rand.added_com_range_z
            rand_com = np.random.uniform([rng_com_x[0], rng_com_y[0], rng_com_z[0]], [rng_com_x[1], rng_com_y[1], rng_com_z[1]], size=(3, ))
            props[1].com += gymapi.Vec3(*rand_com)
        else:
            rand_com = np.zeros(3)

        mass_params = np.concatenate([rand_mass, rand_com, gripper_rand_mass])
        return props, mass_params
    
    def _box_process_rigid_body_props(self, props, env_id):
        if self.cfg.box.randomize_base_mass:
            rng_mass = self.cfg.box.added_mass_range
            rand_mass = np.random.uniform(rng_mass[0], rng_mass[1], size=(1, ))
            props[0].mass += rand_mass
        else:
            rand_mass = np.zeros(1)
        
        return props, rand_mass
    
    def _table_process_rigid_body_props(self, props, env_id):
        if self.cfg.table.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.table.friction_range
                num_buckets = 1000
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]
            props[0].friction = self.friction_coeffs[env_id]
            props[0].restitution = 0.0
        return props

    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 1000
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        
        else:
            if env_id == 0:
                self.friction_coeffs = torch.ones((self.num_envs, 1, 1)) 
        
        return props

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        self.action_scale = torch.tensor(self.cfg.control.action_scale, device=self.device)

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        jacobian_tensor = self.gym.acquire_jacobian_tensor(self.sim, "robot_dog")
        force_sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.force_sensor_tensor = gymtorch.wrap_tensor(force_sensor_tensor).view(self.num_envs, 4, 6)
        self._root_states = gymtorch.wrap_tensor(actor_root_state).view(self.num_envs, 1, 13) # 1 actor
        self.root_states = self._root_states[:, 0, :]
        # self.box_root_state = self._root_states[:, 1, :]            
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_pos_wo_gripper = self.dof_pos[:, :-self.cfg.env.num_gripper_joints]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]
        self.dof_vel_wo_gripper = self.dof_vel[:, :-self.cfg.env.num_gripper_joints]
        self.base_quat = self.root_states[:, 3:7]
        self.base_pos = self.root_states[:, :3]
        self.arm_base_offset = torch.tensor(self._load_arm_base_offset_from_urdf(), device=self.device, dtype=torch.float).repeat(self.num_envs, 1)
        # self.yaw_ema = euler_from_quat(self.base_quat)[2]
        base_yaw = euler_from_quat(self.base_quat)[2]
        self.base_yaw_euler = torch.cat([torch.zeros(self.num_envs, 2, device=self.device), base_yaw.view(-1, 1)], dim=1)
        self.base_yaw_quat = quat_from_euler_xyz(torch.tensor(0), torch.tensor(0), base_yaw)

        self.obs_history_buf = torch.zeros(self.num_envs, self.cfg.env.history_len, self.cfg.env.num_proprio, device=self.device, dtype=torch.float)
        self.action_history_buf = torch.zeros(self.num_envs, self.action_delay + 2, self.num_actions, device=self.device, dtype=torch.float)

        self._contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self.contact_forces = self._contact_forces[:, :-1, :]
        # self.box_contact_force = self._contact_forces[:, -1, :]
        # self.box_rigid_body_state = self._rigid_body_state[:, -2, :]
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, self.num_bodies, 13)
        self.rigid_body_state = self._rigid_body_state[:, :, :]
        self.jacobian_whole = gymtorch.wrap_tensor(jacobian_tensor)
        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:,
                               self.feet_indices,
                               7:10]
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices,
                              0:3]

        # ee info
        self.ee_pos = self.rigid_body_state[:, self.gripper_idx, :3]
        self.ee_orn = self.rigid_body_state[:, self.gripper_idx, 3:7]
        self.ee_vel = self.rigid_body_state[:, self.gripper_idx, 7:]
        self.ee_j_eef = self.jacobian_whole[:, self.gripper_idx, :6, -(6 + self.cfg.env.num_gripper_joints):-self.cfg.env.num_gripper_joints]


        # box info & target_ee info
        # self.box_pos = self.box_root_state[:, 0:3]
        self.grasp_offset = self.cfg.arm.grasp_offset
        self.init_target_ee_base = torch.tensor(self.cfg.arm.init_target_ee_base, device=self.device).unsqueeze(0)

        self.ee_goals = EeGoalManager(self)
        self.ee_goals.init_buffers()
        self.manip = ManipulabilityTracker(self)
        self.manip.init_buffers()
        self.arm = ArmController(self)
        self.arm.init_buffers()

        if self.cfg.debug.test_control:
            self.errors2log = []
            self.recorded_num = -2
        


        #noise
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.add_noise = self.cfg.noise.add_noise

        print('------------------------------------------------------')
        print(f'root_states shape: {self.root_states.shape}')
        print(f'dof_state shape: {self.dof_state.shape}')
        print(f'force_sensor_tensor shape: {self.force_sensor_tensor.shape}')
        print(f'contact_forces shape: {self.contact_forces.shape}')
        print(f'rigid_body_state shape: {self.rigid_body_state.shape}')
        print(f'jacobian_whole shape: {self.jacobian_whole.shape}')
        # print(f'box_root_state shape: {self.box_root_state.shape}')
        # print(f'box_contact_force shape: {self.box_contact_force.shape}')
        # print(f'box_rigid_body_state shape: {self.box_rigid_body_state.shape}')
        print('------------------------------------------------------')
        
        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.extras["episode"] = {}
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_torques, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_torques, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.last_torques = torch.zeros_like(self.torques)

        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
                                           device=self.device, requires_grad=False, )[:self.cfg.commands.num_commands]
        # Refer to <walk these ways>, only useful when `self.cfg.env.observe_gait_commands` is True
        self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                  requires_grad=False, )
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.doubletime_clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                   requires_grad=False)
        self.halftime_clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                 requires_grad=False)
        
        # self.target_ee = torch.zeros(self.num_envs, self.cfg.target_ee.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # ee x, ee y, ee z
        self.gripper_torques_zero = torch.zeros(self.num_envs, self.cfg.env.num_gripper_joints, device=self.device)

        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
        
        for i in range(self.num_torques):
            name = self.dof_names[i]
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    raise Exception(f"PD gain of joint {name} were not defined, setting them to zero")
        # self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.default_dof_pos_wo_gripper = self.default_dof_pos[:-self.cfg.env.num_gripper_joints]
        
        self.global_steps = 0
                

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.root_states[env_ids, :2] += torch_rand_float(-self.cfg.init_state.origin_perturb_range, self.cfg.init_state.origin_perturb_range, (len(env_ids), 2), device=self.device) # xy position within 1m of the center

            
        # base orientation
        rand_yaw = self.cfg.init_state.rand_yaw_range*torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
        quat = quat_from_euler_xyz(0*rand_yaw, 0*rand_yaw, rand_yaw) 
        self.root_states[env_ids, 3:7] = quat[:, :]  
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-self.cfg.init_state.init_vel_perturb_range, self.cfg.init_state.init_vel_perturb_range, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel

        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_states))
        self.gym.refresh_actor_root_state_tensor(self.sim)

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.root_states[:, 7:9] = torch.where(
            self.commands.sum(dim=1).unsqueeze(-1) == 0,
            self.root_states[:, 7:9] * 2.5,
            self.root_states[:, 7:9]
        )
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_states))

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.8, 1.2, (len(env_ids), self.num_dofs), device=self.device)
        self.dof_vel[env_ids] = 0.

        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        
    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """

        if self.cfg.env.teleop_mode:
            return
        # if torch.rand(1).item() > 0.5:

        if self.global_steps < 5000 * 24 and not self.cfg.debug.vl_train: # 5000 can learn forward
            self.commands[env_ids, 0] = torch_rand_float(0, self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            # self.commands[env_ids, 1] = torch.zeros((len(env_ids)), device=self.device)
        else:                
            self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.debug.test_control:
            self.commands[env_ids, 0] = 0.0*torch.ones((len(env_ids)), device=self.device)
            self.commands[env_ids, 1] = 0.0*torch.ones((len(env_ids)), device=self.device)
            self.commands[env_ids, 2] = 0.0*torch.ones((len(env_ids)), device=self.device)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        # self.commands[env_ids, 0] = torch.ones((len(env_ids)), device=self.device)
        # self.commands[env_ids, 1] = torch.ones((len(env_ids)), device=self.device)
        # self.commands[env_ids, 2] = torch.zeros((len(env_ids)), device=self.device)
        # else:
        #     self.commands[env_ids, 0] = torch.zeros((len(env_ids)), device=self.device)
        #     self.commands[env_ids, 1] = torch.zeros((len(env_ids)), device=self.device)
        #     self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        # set small commands to zero
        self.commands[env_ids, :] *= (torch.logical_or(torch.norm(self.commands[env_ids, :2]) > self.cfg.commands.lin_vel_x_clip, torch.abs(self.commands[env_ids, 2]) > self.cfg.commands.ang_vel_yaw_clip)).unsqueeze(1)

    def log_errors(self, env_ids):
        """Record velocity errors"""
        # 计算各个方向的误差
        x_error = abs(self.commands[env_ids, 0] - self.base_lin_vel[env_ids, 0]).item()
        y_error = abs(self.commands[env_ids, 1] - self.base_lin_vel[env_ids, 1]).item()
        yaw_error = abs(self.commands[env_ids, 2] - self.base_ang_vel[env_ids, 2]).item()

        ee_pos_error = torch.norm(self.ee_pos[env_ids] - self.curr_ee_goal_cart_world[env_ids]).item()
        ee_orn_error = torch.norm(orientation_error(self.ee_goal_orn_quat, self.ee_orn / torch.norm(self.ee_orn, dim=-1).unsqueeze(-1))[env_ids]).item()
        # 保存到内存
        self.errors2log.append((x_error, y_error, yaw_error, ee_pos_error, ee_orn_error))

        # 同时写入文件（实时保存）
        # self._write_to_csv(x_error, y_error, yaw_error)
        
        # return x_error, y_error, yaw_error
    
    def save_csv_final(self):
        """ Save errors to CSV file """
        error_save_path = "errors.csv"
        self.recorded_num += 1
        with open(error_save_path, 'a', newline='') as f:
            writer = csv.writer(f)
            # head
            writer.writerow(['x_velocity_error', 'y_velocity_error', 'yaw_velocity_error', 'ee_position_error', 'ee_orientation_error'])
            # write data
            writer.writerows(self.errors2log)
        print(self.recorded_num)
        print(f"Successfully saved {len(self.errors2log)} records to {error_save_path}")
        self.errors2log = []

    def _step_contact_targets(self):
        if self.cfg.env.observe_gait_commands:
            frequencies = self.cfg.env.frequencies
            phases = 0.5
            offsets = 0
            bounds = 0
            durations = 0.5
            self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)
            self.gait_indices[~self._get_walking_cmd_mask()] = 0

            foot_indices = [self.gait_indices + phases + offsets + bounds,
                            self.gait_indices + offsets,
                            self.gait_indices + bounds,
                            self.gait_indices + phases]

            self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)

            for idxs in foot_indices:
                stance_idxs = torch.remainder(idxs, 1) < durations
                swing_idxs = torch.remainder(idxs, 1) > durations

                idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations)
                idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations) * (
                            0.5 / (1 - durations))

            self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
            self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
            self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
            self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

            self.doubletime_clock_inputs[:, 0] = torch.sin(4 * np.pi * foot_indices[0])
            self.doubletime_clock_inputs[:, 1] = torch.sin(4 * np.pi * foot_indices[1])
            self.doubletime_clock_inputs[:, 2] = torch.sin(4 * np.pi * foot_indices[2])
            self.doubletime_clock_inputs[:, 3] = torch.sin(4 * np.pi * foot_indices[3])

            self.halftime_clock_inputs[:, 0] = torch.sin(np.pi * foot_indices[0])
            self.halftime_clock_inputs[:, 1] = torch.sin(np.pi * foot_indices[1])
            self.halftime_clock_inputs[:, 2] = torch.sin(np.pi * foot_indices[2])
            self.halftime_clock_inputs[:, 3] = torch.sin(np.pi * foot_indices[3])

            # von mises distribution
            kappa = self.cfg.rewards.kappa_gait_probs
            smoothing_cdf_start = torch.distributions.normal.Normal(0,
                                                                    kappa).cdf  # (x) + torch.distributions.normal.Normal(1, kappa).cdf(x)) / 2

            smoothing_multiplier_FL = (smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5)) +
                                       smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1) * (
                                               1 - smoothing_cdf_start(
                                           torch.remainder(foot_indices[0], 1.0) - 0.5 - 1)))
            smoothing_multiplier_FR = (smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5)) +
                                       smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1) * (
                                               1 - smoothing_cdf_start(
                                           torch.remainder(foot_indices[1], 1.0) - 0.5 - 1)))
            smoothing_multiplier_RL = (smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5)) +
                                       smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 1) * (
                                               1 - smoothing_cdf_start(
                                           torch.remainder(foot_indices[2], 1.0) - 0.5 - 1)))
            smoothing_multiplier_RR = (smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5)) +
                                       smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 1) * (
                                               1 - smoothing_cdf_start(
                                           torch.remainder(foot_indices[3], 1.0) - 0.5 - 1)))

            self.desired_contact_states[:, 0] = smoothing_multiplier_FL
            self.desired_contact_states[:, 1] = smoothing_multiplier_FR
            self.desired_contact_states[:, 2] = smoothing_multiplier_RL
            self.desired_contact_states[:, 3] = smoothing_multiplier_RR
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        command_env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        if self.cfg.experiment.use_cmd and len(command_env_ids) > 0:
            self._resample_commands(command_env_ids)
        self._step_contact_targets()

        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.push_interval == 0):
            self._push_robots()
    
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure
        Args:
            cfg (Dict): Environment config file
        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """

        if self.num_actions not in (12, 20):
            raise NotImplementedError(f"Noise scale is only implemented for action space of 12 or 20 (got {self.num_actions})")

        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        idx = 0

        # Body orientation (dim 2)
        noise_vec[idx:idx+2] = 0  # Assuming no noise for body orientation
        idx += 2

        # Angular velocity (dim 3)   
        noise_vec[idx:idx+3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        idx += 3

        # DOF positions (dim 12) b1
        noise_vec[idx:idx+12] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        idx += 12

        # DOF positions (dim 6) z1
        noise_vec[idx:idx+6] = 0
        idx += 6

        # DOF velocities (dim 12)
        noise_vec[idx:idx+12] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        idx += 12

        # DOF velocities (dim 6)
        noise_vec[idx:idx+6] = 0
        idx += 6

        # Action history (dim 12)
        noise_vec[idx:idx+12] = 0  # Assuming no noise for action history
        idx += 12

        # Foot contacts (dim 4)
        noise_vec[idx:idx+4] = 0  # Assuming no noise for foot contacts
        idx += 4

        # Commands (dim 3)
        noise_vec[idx:idx+3] = 0  # Assuming no noise for commands
        idx += 3

        # End-effector goal position (dim 3)
        noise_vec[idx:idx+3] = 0  # Assuming no noise for end-effector goal position
        idx += 3

        # End-effector goal orientation (dim 3)
        noise_vec[idx:idx+3] = 0  # Assuming no noise for end-effector goal orientation
        idx += 3

        if self.cfg.terrain.measure_heights:
            noise_vec[48:235] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements

        return noise_vec
    
    def _reindex_feet(self, vec):
        # raisim_order = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
        # ig_order = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
        if self.cfg.env.reorder_dofs:
            return vec[:, [1, 0, 3, 2]]
        return vec

    def _reindex_all(self, vec):
        return torch.hstack((vec[:, [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]], vec[:, 12:]))
    
    def _reindex_dog(self, vec):
        return vec[:, [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]]
    
    def _get_body_orientation(self, return_yaw=False):
        r, p, y = euler_from_quat(self.base_quat)
        body_angles = torch.stack([r, p, y], dim=-1)

        if not return_yaw:
            return body_angles[:, :-1]
        else:
            return body_angles

    def _draw_collision_bbox(self):

        center = self.ee_goal_center_offset
        bbox0 = center + self.collision_upper_limits
        bbox1 = center + self.collision_lower_limits
        bboxes = torch.stack([bbox0, bbox1], dim=1)
        sphere_geom = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(1, 1, 0))

        for i in range(self.num_envs):
            bbox_geom = gymutil.WireframeBBoxGeometry(bboxes[i], None, color=(1, 0, 0))
            quat = self.base_yaw_quat[i]
            r = gymapi.Quat(quat[0], quat[1], quat[2], quat[3])
            pose0 = gymapi.Transform(gymapi.Vec3(self.root_states[i, 0], self.root_states[i, 1], 0), r=r)
            gymutil.draw_lines(bbox_geom, self.gym, self.viewer, self.envs[i], pose=pose0) 

    def _draw_ee_goal_curr(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        sphere_geom = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(1, 1, 0))

        sphere_geom_3 = gymutil.WireframeSphereGeometry(0.05, 16, 16, None, color=(0, 1, 1))
        upper_arm_pose = self.ee_goals.get_spherical_center()

        sphere_geom_2 = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(0, 0, 1))
        ee_pose = self.rigid_body_state[:, self.gripper_idx, :3]

        # sphere_geom_origin = gymutil.WireframeSphereGeometry(0.1, 8, 8, None, color=(0, 1, 0))
        sphere_pose = gymapi.Transform(gymapi.Vec3(0, 0, 0), r=None)
        # gymutil.draw_lines(sphere_geom_origin, self.gym, self.viewer, self.envs[0], sphere_pose)

        axes_geom = gymutil.AxesGeometry(scale=0.2)

        for i in range(self.num_envs):
            sphere_pose = gymapi.Transform(gymapi.Vec3(self.curr_ee_goal_cart_world[i, 0], self.curr_ee_goal_cart_world[i, 1], self.curr_ee_goal_cart_world[i, 2]), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 
            
            sphere_pose_2 = gymapi.Transform(gymapi.Vec3(ee_pose[i, 0], ee_pose[i, 1], ee_pose[i, 2]), r=None)
            gymutil.draw_lines(sphere_geom_2, self.gym, self.viewer, self.envs[i], sphere_pose_2) 

            # sphere_pose_3 = gymapi.Transform(gymapi.Vec3(upper_arm_pose[i, 0], upper_arm_pose[i, 1], upper_arm_pose[i, 2]), r=None)
            # gymutil.draw_lines(sphere_geom_3, self.gym, self.viewer, self.envs[i], sphere_pose_3) 

            pose = gymapi.Transform(gymapi.Vec3(self.curr_ee_goal_cart_world[i, 0], self.curr_ee_goal_cart_world[i, 1], self.curr_ee_goal_cart_world[i, 2]), 
                                    r=gymapi.Quat(self.ee_goal_orn_quat[i, 0], self.ee_goal_orn_quat[i, 1], self.ee_goal_orn_quat[i, 2], self.ee_goal_orn_quat[i, 3]))
            gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[i], pose)

    def _draw_ee_goal_traj(self):
        sphere_geom = gymutil.WireframeSphereGeometry(0.005, 8, 8, None, color=(1, 0, 0))
        sphere_geom_yellow = gymutil.WireframeSphereGeometry(0.01, 16, 16, None, color=(1, 1, 0))

        t = torch.linspace(0, 1, 10, device=self.device)[None, None, None, :]
        ee_target_all_sphere = torch.lerp(self.ee_start_sphere[..., None], self.ee_goal_sphere[..., None], t).squeeze(0)
        ee_target_all_cart_world = torch.zeros_like(ee_target_all_sphere)
        for i in range(10):
            ee_target_cart = sphere2cart(ee_target_all_sphere[..., i])
            if self.cfg.experiment.fix_sample:
                ee_target_all_cart_world[..., i] = quat_apply(self.base_yaw_quat_fixed, ee_target_cart)
            else:
                ee_target_all_cart_world[..., i] = quat_apply(self.base_yaw_quat, ee_target_cart)
        if self.cfg.experiment.fix_sample:
            ee_target_all_cart_world += self.centers_fixed[:, :, None]
        else:
            ee_target_all_cart_world += self.ee_goals.get_spherical_center()[:, :, None]
        for i in range(self.num_envs):
            for j in range(10):
                pose = gymapi.Transform(gymapi.Vec3(ee_target_all_cart_world[i, 0, j], ee_target_all_cart_world[i, 1, j], ee_target_all_cart_world[i, 2, j]), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], pose)

    def _draw_feet_target(self):
        sphere_geom_left = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(1, 1, 0))
        sphere_geom_right = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(0, 1, 1))
        for i in range(self.num_envs):
            sphere_pose_left = gymapi.Transform(gymapi.Vec3(
                self.left_foot_goal_pos_world[i, 0], self.left_foot_goal_pos_world[i, 1], self.left_foot_goal_pos_world[i, 2]), r=None)
            gymutil.draw_lines(sphere_geom_left, self.gym, self.viewer, self.envs[i], sphere_pose_left) 
            sphere_pose_right = gymapi.Transform(gymapi.Vec3(
                self.right_foot_goal_pos_world[i, 0], self.right_foot_goal_pos_world[i, 1], self.right_foot_goal_pos_world[i, 2]), r=None)
            gymutil.draw_lines(sphere_geom_right, self.gym, self.viewer, self.envs[i], sphere_pose_right) 

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        actions_scaled = actions * self.motor_strength * self.action_scale
        # actions_scaled[:, :12] = torch.tensor([
        #     0.02, -0.16, -0.002, 0.001, -0.137, -0.022,  0.032, -0.157, -0.018,  0.013, -0.124, -0.054], device=self.device).repeat(self.num_envs, 1)
        default_torques = self.p_gains * (actions_scaled + self.default_dof_pos_wo_gripper - self.dof_pos_wo_gripper) - self.d_gains * self.dof_vel_wo_gripper
        default_torques[:, -6:] = 0
        torques = torch.cat([default_torques, self.gripper_torques_zero], dim=-1)

        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _get_walking_cmd_mask(self, env_ids=None, return_all=False):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if self.cfg.experiment.use_cmd:
            walking_mask0 = torch.abs(self.commands[env_ids, 0]) > self.cfg.commands.lin_vel_x_clip
            walking_mask1 = torch.abs(self.commands[env_ids, 1]) > self.cfg.commands.lin_vel_y_clip
            walking_mask2 = torch.abs(self.commands[env_ids, 2]) > self.cfg.commands.ang_vel_yaw_clip
        else:
            walking_mask0 = torch.abs(self.dvel_b_local[env_ids, 0]) > self.cfg.commands.lin_vel_x_clip
            walking_mask1 = torch.abs(self.dvel_b_local[env_ids, 1]) > self.cfg.commands.lin_vel_y_clip
            walking_mask2 = torch.abs(self.dvel_b_local[env_ids, 2]) > self.cfg.commands.ang_vel_yaw_clip

        walking_mask = walking_mask0 | walking_mask1 | walking_mask2
        if return_all:
            return walking_mask0, walking_mask1, walking_mask2, walking_mask
        return walking_mask

   # -------------- IssacGym render and viewer functions ----------------

    def render_record(self, mode="rgb_array"):
        if self.global_steps % 2 == 0:
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            imgs = []
            for i in range(self.num_envs):
                cam = self._rendering_camera_handles[i]
                root_pos = self.root_states[i, :3].cpu().numpy()
                cam_pos = root_pos + np.array([0, 2, 1])
                self.gym.set_camera_location(cam, self.envs[i], gymapi.Vec3(*cam_pos), gymapi.Vec3(*root_pos))
                
                img = self.gym.get_camera_image(self.sim, self.envs[i], cam, gymapi.IMAGE_COLOR)
                w, h = img.shape
                imgs.append(img.reshape([w, h // 4, 4]))
            return imgs
        return None

    def subscribe_viewer_keyboard_events(self):
        super().subscribe_viewer_keyboard_events()

        if self.cfg.env.teleop_mode:
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_W, "forward")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_S, "reverse")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_A, "turn_left")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_D, "turn_right")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Q, "stop_linear")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_E, "stop_angular")

            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Y, "increase_eef_goal_l")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_H, "decrease_eef_goal_l")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_U, "increase_eef_goal_p")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_J, "decrease_eef_goal_p")  
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_I, "increase_eef_goal_y")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_K, "decrease_eef_goal_y")

            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Z, "increase_eef_goal_dr")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_X, "decrease_eef_goal_dr")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_C, "increse_eef_goal_dp")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "decrease_eef_goal_dp")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_B, "increase_eef_goal_dy")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_N, "decrease_eef_goal_dy")

    def handle_viewer_action_event(self, evt):
        super().handle_viewer_action_event(evt)

        if evt.value <= 0:
            return
        if evt.action == "stop_linear":
            self.commands[:, 0] = 0
        elif evt.action == "forward":
            self.commands[:, 0] += 0.05
        elif evt.action == "reverse":
            self.commands[:, 0] -= 0.05

        if evt.action == "stop_angular":
            self.commands[:, 2] = 0
        if evt.action == "turn_left":
            self.commands[:, 2] += 0.05
        elif evt.action == "turn_right":
            self.commands[:, 2] -= 0.05




        # Sphere position
        # if evt.action == "increase_eef_goal_l":
        #     self.curr_ee_goal_sphere[:, 0] += 0.05
        # elif evt.action == "decrease_eef_goal_l":
        #     self.curr_ee_goal_sphere[:, 0] -= 0.05

        # if evt.action == "increase_eef_goal_p":
        #     self.curr_ee_goal_sphere[:, 1] += 0.05
        # elif evt.action == "decrease_eef_goal_p":
        #     self.curr_ee_goal_sphere[:, 1] -= 0.05

        # if evt.action == "increase_eef_goal_y":
        #     self.curr_ee_goal_sphere[:, 2] += 0.05
        # elif evt.action == "decrease_eef_goal_y":
        #     self.curr_ee_goal_sphere[:, 2] -= 0.05

        # cartesian position
        if evt.action == "increase_eef_goal_l":
            self.curr_ee_goal_cart[:, 0] += 0.05
        elif evt.action == "decrease_eef_goal_l":
            self.curr_ee_goal_cart[:, 0] -= 0.05

        if evt.action == "increase_eef_goal_p":
            self.curr_ee_goal_cart[:, 1] += 0.05
        elif evt.action == "decrease_eef_goal_p":
            self.curr_ee_goal_cart[:, 1] -= 0.05

        if evt.action == "increase_eef_goal_y":
            self.curr_ee_goal_cart[:, 2] += 0.05
        elif evt.action == "decrease_eef_goal_y":
            self.curr_ee_goal_cart[:, 2] -= 0.05

        self.curr_ee_goal_sphere = cart2sphere(self.curr_ee_goal_cart)
        
        # orientation
        if evt.action == "increase_eef_goal_dr":
            self.ee_goal_orn_delta_rpy[:, 0] += 0.05
        elif evt.action == "decrease_eef_goal_dr":
            self.ee_goal_orn_delta_rpy[:, 0] -= 0.05

        if evt.action == "increse_eef_goal_dp":
            self.ee_goal_orn_delta_rpy[:, 1] += 0.05
        elif evt.action == "decrease_eef_goal_dp":
            self.ee_goal_orn_delta_rpy[:, 1] -= 0.05
        
        if evt.action == "increase_eef_goal_dy":
            self.ee_goal_orn_delta_rpy[:, 2] += 0.05
        elif evt.action == "decrease_eef_goal_dy":
            self.ee_goal_orn_delta_rpy[:, 2] -= 0.05

