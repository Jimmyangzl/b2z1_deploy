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

from legged_gym_wbc.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np

class B2Z1WBCRoughCfg( LeggedRobotCfg ):
    class experiment:
        fix_arm = False
        use_cmd = False
        fix_sample = False
        # When True (and fix_sample=False): sample EE goals in the live Z1 arm-base frame (base_quat).
        # When False: sample in the live base-yaw / spherical-center frame (stage1-style).
        base_sample = False
        # Fraction of goals that snap (no interpolation). Applies to base_sample and yaw-frame sampling.
        base_sample_far_ratio = 0.0 # 0.2
        # Among non-far goals, fraction that end at default EE. Applies to base_sample and yaw-frame.
        # Yaw-frame: default_ee_cart_arm_base is mapped into the spherical-center / yaw frame.
        base_sample_default_ratio = 0.2 # 0.2
        # World-frame EE height limits for base_sample goals [m].
        base_sample_z_range = [0.35, 0.8]
        # RFM dvel_b_local XY amplitude [m/s] (unit direction × amp). Far goals use the larger value.
        dvel_xy_amp = 0.15
        dvel_xy_amp_far = 0.3
        far_sample = False
        fine_sample = False
        # Probabilities for SPHERE / BACK / ROT when fine_sample is on (need not sum to 1; normalized at use).
        fine_sample_ratios = [1.0, 0.0, 0.0]
        use_rfm = True
        use_prio_reward = True
        flag_inference = True
        tele_op = False

    class debug:
        vl_train = False
        test_control = False
        test_vel = False

    class goal_ee:
        num_commands = 3
        traj_time = [3, 5]
        hold_time = [1.0, 2.0]
        # hold_time = [1, 3]
        collision_upper_limits = [0.1, 0.2, -0.05]
        collision_lower_limits = [-0.8, -0.2, -0.7]
        underground_limit = -0.7
        num_collision_check_samples = 10
        command_mode = 'sphere'
        arm_induced_pitch = 0.38 # Added to -pos_p (negative goal pitch) to get default eef orn_p

        class sphere_center:
            # x_offset = 0.3 # Relative to base
            x_offset = 0.15 # Relative to base
            # x_offset = 0.0
            # x_offset = 0.35 # Relative to base
            y_offset = 0 # Relative to base
            # z_invariant_offset = 0.7 # Relative to terrain
            z_invariant_offset = 0.7

        class ranges:
            init_pos_start = [0.5, np.pi/8, 0]
            init_pos_end = [0.7, 0, 0]
            # init_pos_start = [0.0, 0, 0]
            # init_pos_end = [0.1, 0, 0]
            # pos_l = [0.4, 0.95] # USE this when using cmds
            pos_l = [0.5, 1.0]
            # pos_l = [0.7, 1.5]
            # pos_l = [1., 3.0]
            # pos_p = [-1 * np.pi / 2.5, 1 * np.pi / 3]
            pos_p = [-1 * np.pi / 6, 1 * np.pi / 12]
            pos_y = [-np.pi/3 - 0.2, np.pi/3 + 0.2]
            # pos_y = [-2*np.pi/3, 2*np.pi/3] # 3*np.pi/5
            # pos_y = [-np.pi/3.0, np.pi/3.0]

            # base_sample far subset (no interpolation): separate sphere ranges
            base_far_pos_l = [2.0, 2.5]
            base_far_pos_p = [-1 * np.pi / 12, 1 * np.pi / 12]
            base_far_pos_y = [-np.pi/12.0, np.pi/12.0]

            back_pos_l = [0.3, 0.5]
            
            delta_orn_r = [-0.5, 0.5]
            # delta_orn_p = [-0.5, 0.5]
            delta_orn_p = [-0.1, 0.3]
            delta_orn_y = [-0.5, 0.5]

            final_tracking_ee_reward = 0.55

        sphere_error_scale = [1, 1, 1]#[1 / (ranges.final_pos_l[1] - ranges.final_pos_l[0]), 1 / (ranges.final_pos_p[1] - ranges.final_pos_p[0]), 1 / (ranges.final_pos_y[1] - ranges.final_pos_y[0])]
        orn_error_scale = [1, 1, 1]#[2 / np.pi, 2 / np.pi, 2 / np.pi]

    class noise:
        add_noise = False
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1   

    class commands:
        curriculum = True
        num_commands = 3
        # resampling_time = 3. # time before command are changed[s]
        resampling_time = 4.

        # lin_vel_x_schedule = [0, 2.0]
        # lin_vel_y_schedule = [0, 2.0]
        # ang_vel_yaw_schedule = [0, 1]
        # tracking_ang_vel_yaw_schedule = [0, 1]

        # ang_vel_yaw_clip = 0.5
        # lin_vel_x_clip = 0.8
        # lin_vel_y_clip = 0.8
        ang_vel_yaw_clip = 0.2
        lin_vel_x_clip = 0.1
        lin_vel_y_clip = 0.1
        lin_vel_clip = 0.1

        class ranges:
            # lin_vel_x = [-0.8, 0.8] # min max [m/s]
            # lin_vel_y = [-0.8, 0.8] # min max [m/s]
            # ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            lin_vel_x = [-0.3, 0.3] # min max [m/s]
            lin_vel_y = [-0.3, 0.3] # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]    # min max [rad/s]

    class normalization:
        class obs_scales:
            lin_vel = 1.0
            ang_vel =  1.0
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 20.
        clip_lmbdas = 2.

    class env:
        num_envs = 4096 #4096
        num_actions = 12  # legs only; arm pose via IK
        # num_actions = 12 + 6 + 2 #CAUTION
        # num_actions = 12 + 6 #CAUTION
        num_torques = 12 + 6
        action_delay = 3  # -1 for no delay
        num_gripper_joints = 1
        # num_proprio = 2 + 3 + 18 + 18 + 12 + 4 + 3 + 3 + 3 + 3
        num_proprio = 2 + 3 + 18 + 18 + 12 + 4 + 3 + 3 + 3 + 3 + 1
        num_priv = 5 + 1 + 12
        # num_priv = 0
        history_len = 10
        num_observations = num_proprio * (history_len+1) + num_priv
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds
        reorder_dofs = False
        teleop_mode = False # Overriden in teleop.py. When true, commands come from keyboard
        record_video = False
        stand_by = False
        observe_gait_commands = True
        frequencies = 2
        
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.55] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            # 'FL_hip_joint': 0.2,   # [rad]
            'FL_hip_joint': 0.0,   # [rad]
            'FL_thigh_joint': 0.8,     # [rad]
            'FL_calf_joint': -1.5,   # [rad]

            # 'RL_hip_joint': 0.2,   # [rad]
            'RL_hip_joint': 0.0,   # [rad]
            'RL_thigh_joint': 0.8,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]

            # 'FR_hip_joint': -0.2 ,  # [rad]
            'FR_hip_joint': -0.0 ,  # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'FR_calf_joint': -1.5,  # [rad]

            # 'RR_hip_joint': -0.2,   # [rad]
            'RR_hip_joint': -0.0,   # [rad]
            'RR_thigh_joint': 0.8,   # [rad]
            'RR_calf_joint': -1.5,    # [rad]
            # 'z1_waist': 0.0,
            # 'z1_shoulder': 1.48,
            # 'z1_elbow': -0.63,
            # 'z1_wrist_angle': -0.84,
            # 'z1_forearm_roll': 0.0,
            # 'z1_wrist_rotate': 1.57,#0.0,
            # 'z1_jointGripper': -0.785,
            'z1_waist': 0.0,
            'z1_shoulder': 1.8,
            # 'z1_shoulder': 2*np.pi/3,
            'z1_elbow': -0.9,
            # 'z1_elbow': -2*np.pi/3,
            'z1_wrist_angle': -0.9,
            'z1_forearm_roll': 0.0,
            # 'z1_wrist_rotate': 1.57,#0.0,
            'z1_wrist_rotate': 0.0,
            'z1_jointGripper': -0.785,
        }
        # rand_yaw_range = np.pi/2
        rand_yaw_range = 0
        # rand_yaw_range = np.pi/12
        # origin_perturb_range = 0.5
        # init_vel_perturb_range = 0.1
        # origin_perturb_range = 0.1
        # init_vel_perturb_range = 0.1
        origin_perturb_range = 0.0
        init_vel_perturb_range = 0.0

    class control:
        # Per-leg-joint PD gains [N*m/rad] / [N*m*s/rad]. Keys must be substrings of dof names.
        # stiffness = {
        #     'FL_hip_joint': 200,
        #     'FL_thigh_joint': 200,
        #     'FL_calf_joint': 200,
        #     'FR_hip_joint': 200,
        #     'FR_thigh_joint': 200,
        #     'FR_calf_joint': 200,
        #     'RL_hip_joint': 200,
        #     'RL_thigh_joint': 200,
        #     'RL_calf_joint': 200,
        #     'RR_hip_joint': 200,
        #     'RR_thigh_joint': 200,
        #     'RR_calf_joint': 200,
        #     'z1': 100,
        # }
        # damping = {
        #     'FL_hip_joint': 10.0,
        #     'FL_thigh_joint': 10.0,
        #     'FL_calf_joint': 10.0,
        #     'FR_hip_joint': 10.0,
        #     'FR_thigh_joint': 10.0,
        #     'FR_calf_joint': 10.0,
        #     'RL_hip_joint': 10.0,
        #     'RL_thigh_joint': 10.0,
        #     'RL_calf_joint': 10.0,
        #     'RR_hip_joint': 10.0,
        #     'RR_thigh_joint': 10.0,
        #     'RR_calf_joint': 10.0,
        #     'z1': 5.0,
        # }
        stiffness = {'joint': 200, 'z1': 5}  # [N*m/rad] # Kp: 80, 150, 200
        damping = {'joint': 10.0, 'z1': 0.5}     # [N*m*s/rad]

        adaptive_arm_gains = False
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = [0.4, 0.45, 0.45] * 2 + [0.4, 0.45, 0.45] * 2 + [2.1, 0.6, 0.6, 0, 0, 0]
        # Hard clamp on desired joint positions [rad]. None disables.
        # Defaults match b2z1.urdf <limit lower/upper>. Gripper omitted (held at default).
        hard_dof_pos_limit = {
            'FL_hip_joint': (-0.2, 0.2),
            'FL_thigh_joint': (-0.94, 1.0),
            'FL_calf_joint': (-2.82, -1.0),
            'FR_hip_joint': (-0.2, 0.2),
            'FR_thigh_joint': (-0.94, 1.0),
            'FR_calf_joint': (-2.82, -1.0),
            'RL_hip_joint': (-0.2, 0.2),
            'RL_thigh_joint': (-0.94, 1.0),
            'RL_calf_joint': (-2.82, -1.0),
            'RR_hip_joint': (-0.2, 0.2),
            'RR_thigh_joint': (-0.94, 1.0),
            'RR_calf_joint': (-2.82, -1.0),
            'z1_waist': (-2.5, 2.5),
            # 'z1_shoulder': (0.0, 3.141592653589793),
            'z1_shoulder': (0.0, 2.8),
            # 'z1_elbow': (-3.0, 0.0),
            'z1_elbow': (-2.5, 0.0),
            # 'z1_wrist_angle': (-1.7453292519943295, 1.5707963267948966),
            'z1_wrist_angle': (-1.5, 1.5),
            # 'z1_forearm_roll': (-1.7278759594743864, 1.7278759594743864),
            'z1_forearm_roll': (-1.5, 1.5),
            # 'z1_wrist_rotate': (-2.792526803190927, 2.792526803190927),
            'z1_wrist_rotate': (-2.5, 2.5),
        }
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        torque_supervision = False

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/b1z1/urdf/b2z1.urdf'
        foot_name = "foot"
        gripper_name = "ee_gripper_link" #"gripperMover"
        penalize_contacts_on = ["thigh", "floating_base", "calf"]
        terminate_after_contacts_on = []
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        collapse_fixed_joints = True # Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False
    
    class box:
        box_size = 0.1
        randomize_base_mass = True
        added_mass_range = [-0.001, 0.050]
        box_env_origins_x = 0
        box_env_origins_y_range = [0.1, 0.3]
        box_env_origins_z = box_size / 2 + 0.16

    class table:
        table_dimz = 0.02
        table_dimx = 0.6
        table_dimy = 1.
        pos_x_from_origin = 1.2
        pos_y_from_origin = 0.
        pos_z_from_origin = 0.5
        randomize_friction = True
        friction_range = [0.3, 3.0]
        randomize_height = True
        table_height_range = [0.3, 1.]
        depth_feet_pos = 0.01 # The distance from the desired feet position to the edge of the table 
        randomize_feet_depth_pos = True
        feet_depth_pos_range = [0.01, 0.05] # The added distance to depth_feet_pos
        half_distance_feet_pos = 0.22 # The distance between two desired feet position
        randomize_feet_distance = True
        feet_distance_range = [-0.05, 0.05]

        wall_dimz = 2.0
        wall_dimx = 0.6
        wall_dimy = 1.
        feet_wall_height = 0.7
        randomize_feet_wall_height = True
        feet_wall_height_range = [-0.2, 0.2]
    
    class arm:
        init_target_ee_base = [0.5, 0.0, 0.4]
        # EE cart / orn (rpy) in arm-base frame at default_joint_angles (base_sample_default_ratio).
        default_ee_cart_arm_base = [0.5, 0.0, 0.4]
        default_ee_orn_arm_base = [0.0, 0.0, 0.0]
        grasp_offset = 0.08
        osc_kp = np.array([100, 100, 100, 30, 30, 30])
        osc_kd = 2 * (osc_kp ** 0.5)

    class domain_rand:
        observe_priv = True
        # observe_priv = False
        randomize_friction = True
        friction_range = [0.4, 1.5] # [0.5, 3.0]
        randomize_base_mass = True
        added_mass_range = [0., 15.]
        randomize_base_com = True
        # added_com_range_x = [-0.15, 0.15]
        # added_com_range_y = [-0.15, 0.15]
        # added_com_range_z = [-0.15, 0.15]
        added_com_range_x = [-0.05, 0.05]
        added_com_range_y = [-0.03, 0.03]
        added_com_range_z = [-0.02, 0.02]
        randomize_motor = True
        leg_motor_strength_range = [0.7, 1.3]
        arm_motor_strength_range = [0.7, 1.3]
        randomize_gripper_mass = True
        # gripper_added_mass_range = [0, 0.1]
        gripper_added_mass_range = [0, 2.0]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 0.5
  
    class rewards:
        reward_container_name = "maniploco_rewards"   # select the reward container to use

        # -------Common Para. ---------
        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.2  # tracking reward = exp(-error^2/sigma)
        tracking_ee_sigma = 1
        soft_dof_pos_limit = 0.95  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.95
        soft_torque_limit = 0.8
        base_height_target = 0.5
        max_contact_force = 300.  # forces above this value are penalized
        # -------Gait control Para. ---------
        gait_vel_sigma = 0.5
        gait_force_sigma = 0.5
        kappa_gait_probs = 0.07
        # feet_height_target = 0.3
        feet_height_target = 0.05

        feet_aritime_allfeet = True
        feet_height_allfeet = True

        # Scales set to 0 will still be logged (as zero reward and non-zero metric)
        # To not compute and log a given metric, set the scale to None
        class scales:
            # -------Gait control rewards ---------
            # tracking_contacts_shaped_force = -4.0 # Only works when `observing_gait_commands` is true
            # tracking_contacts_shaped_vel = -4.0 # Only works when `observing_gait_commands` is true
            tracking_contacts_shaped_force = 4.0 # Only works when `observing_gait_commands` is true
            tracking_contacts_shaped_vel = 4.0 # Only works when `observing_gait_commands` is true
            
            feet_air_time = 2.0
            # feet_air_time = 0.5
            feet_height = 1.0
            # feet_height = 0.5

            # -------Tracking rewards ----------
            tracking_lin_vel_max = None
            tracking_lin_vel_x_l1 = None
            
            tracking_lin_vel_x_exp = None
            tracking_lin_vel_y = None
            tracking_ang_vel = None # 2.0 use this when using cmd
            tracking_lin_vel = None # 5.0 use this when using cmd

            base_acc = 1.0
            energy_square = None
            # torques = -2.5e-5 
            torques = -2.0e-5 
            stand_still = 10.0
            stand_still_vel = None
            # stand_still = None
            walking_dof = 2.0
            # walking_dof = None
            dof_default_pos = None
            dof_error = None
            alive = 1.0
            # alive = 0.0
            lin_vel_z = -1.5
            # lin_vel_z = None
            roll = -2
            # roll = None

            front_feet_goal = None
            rear_feet_approach = None

            # common rewards
            ang_vel_xy = -0.2
            # Penalize base roll angular velocity ω_x (body frame).
            ang_vel_x = -0.5
            dof_acc = -7.5e-7 
            collision = -10.
            # action_rate = -0.015
            action_rate = -0.25
            dof_pos_limits = -10.0
            delta_torques = -1.0e-7
            # hip_pos = -0.3
            hip_pos = -1.0
            work = -0.003
            feet_jerk = None
            # feet_jerk = None
            # feet_drag = -0.08
            feet_drag = -0.15
            # feet_drag = None
            feet_contact_forces = -0.001
            # feet_contact_forces = None
            orientation = -1.0
            orientation_walking = None
            orientation_standing = None
            # Reward avoiding base pitching upward (negative pitch from euler_from_quat); exp form.
            base_pitch_neg = 5.0
            base_height = -5.0
            # base_height = -0.0
            torques_walking = None
            torques_standing = None
            energy_square = None
            energy_square_walking = None
            energy_square_standing = None
            base_height_walking = None
            base_height_standing = None
            penalty_lin_vel_y = None

            base_pos = None
            manipulability = 1.0 # 1.0 USE None when using cmds
            # tracking_base_vel = 1.0 # 2.0
            regularize_lambda = None
        
        class loco_scales:
            loco_base_height = 1.0
            # tracking_v_base = 5.0
            tracking_v_base_max = 5.0
            tracking_ang_v_base = 2.0
            loco_pitch = -5.0

        class arm_scales:
            arm_termination = None
            tracking_ee_sphere = None
            # tracking_ee_world = 0.8
            # tracking_ee_world = 2.0
            # tracking_ee_orn_quat = 1.0
            # tracking_dvel_b_local = 5.0 # 5.0 Use None when using cmds
            tracking_dvel_b_local_max = 5.0 # 5.0 Use None when using cmds
            tracking_ang_dvel_b_local = 2.0 # 5.0 Use None when using cmds
            tracking_ee_world = 0.8
            tracking_ee_orn_quat = 0.5
            tracking_ee_sphere_walking = None
            tracking_ee_sphere_standing = None
            tracking_ee_cart = None
            arm_orientation = None
            arm_energy_abs_sum = None
            tracking_ee_orn = None
            tracking_ee_orn_ry = None
        
    class viewer:
        pos = [-20, 0, 20]  # [m]
        lookat = [0, 0, -2]  # [m]

    class termination:
        r_threshold = 0.8
        p_threshold = 0.8
        z_threshold = 0.1

    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        hf2mesh_method = "fast"  # grid or fast
        max_error = 0.1 # for fast
        horizontal_scale = 0.05 # [m] influence computation time by a lot
        vertical_scale = 0.002 # [m]
        border_size = 25 # [m]
        # height = [0.00, 0.1] # [0.04, 0.1]
        height = [0.00, 0.001] # [0.04, 0.1]
        # height = [0.00, 0.001]
        gap_size = [0.02, 0.1]
        stepping_stone_distance = [0.02, 0.08]
        downsampled_scale = 0.075
        curriculum = False

        all_vertical = False
        no_flat = True
        
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.

        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)  # spreaded is benifitiall !
        num_cols = 10 # number of terrain cols (types)
        # num_rows= 5 # number of terrain rows (levels)  # spreaded is benifitiall !
        # num_cols = 5 # number of terrain cols (types)
        

        terrain_dict = {"smooth slope": 0., 
                        "rough slope up": 0.,
                        "rough slope down": 0.,
                        "rough stairs up": 0., 
                        "rough stairs down": 0., 
                        "discrete": 0., 
                        "stepping stones": 0.,
                        "gaps": 0., 
                        "rough flat": 1.0,
                        "pit": 0.0,
                        "wall": 0.0}
        terrain_proportions = list(terrain_dict.values())
        # trimesh only:
        slope_treshold = None # slopes above this threshold will be corrected to vertical surfaces
        origin_zero_z = True


class B2Z1WBCRoughCfgPPO(LeggedRobotCfgPPO):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        continue_from_last_std = True
        # init_std = [[0.8, 1.0, 1.0] * 4 + [1.0] * 6]
        # init_std = [[0.8, 1.0, 1.0] * 4 + [1.0] * 8]
        init_std = [[0.8, 1.0, 1.0] * 4]
        actor_hidden_dims = [128]
        critic_hidden_dims = [128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # activation = 'tanh'
        output_tanh = False

        leg_control_head_hidden_dims = [128, 128]
        arm_control_head_hidden_dims = [128, 128]

        priv_encoder_dims = [64, 20]

        num_leg_actions = 12
        num_arm_actions = 0
        # num_arm_actions = 6+2
        # num_arm_actions = 6

        adaptive_arm_gains = B2Z1WBCRoughCfg.control.adaptive_arm_gains
        adaptive_arm_gains_scale = 10.0
        
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.0
        # entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 2e-4 
        # learning_rate = 1e-3 
        schedule = 'fixed' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = None
        max_grad_norm = 1.
        # min_policy_std = [[0.15, 0.25, 0.25] * 4 + [0.2] * 3 + [0.05] * 3]
        # min_policy_std = [[0.15, 0.25, 0.25] * 4 + [0.2] * 3 + [0.05] * 3 + [0.02] * 2]
        min_policy_std = [[0.15, 0.25, 0.25] * 4]

        mixing_schedule=[1.0, 0, 3000] #if not RESUME else [1.0, 0, 1]
        torque_supervision = B2Z1WBCRoughCfg.control.torque_supervision  #alert: also appears above
        torque_supervision_schedule=[0.0, 1000, 1000]
        adaptive_arm_gains = B2Z1WBCRoughCfg.control.adaptive_arm_gains
        # dagger params
        dagger_update_freq = 20
        priv_reg_coef_schedual = [0, 0.1, 3000, 7000] #if not RESUME else [0, 1, 1000, 1000]

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24
        max_iterations = 50000 # number of policy updates
        # logging
        save_interval = 500 # check for potential saves every this many iterations
        experiment_name = 'b2z1_v2'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
