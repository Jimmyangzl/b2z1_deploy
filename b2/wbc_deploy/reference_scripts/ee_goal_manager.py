import numpy as np
import torch
from isaacgym.torch_utils import *


class EeGoalManager:
    """End-effector goal sampling, interpolation, and world-frame tracking."""

    SAMPLE_SPHERE = 0
    SAMPLE_BACK = 1
    SAMPLE_ROT = 2

    def __init__(self, env):
        self.env = env

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
        env = self.env
        self.traj_timesteps = torch_rand_float(
            self.cfg.goal_ee.traj_time[0], self.cfg.goal_ee.traj_time[1],
            (self.num_envs, 1), device=self.device,
        ).squeeze(1) / env.dt
        self.traj_total_timesteps = self.traj_timesteps + torch_rand_float(
            self.cfg.goal_ee.hold_time[0], self.cfg.goal_ee.hold_time[1],
            (self.num_envs, 1), device=self.device,
        ).squeeze(1) / env.dt
        self.goal_timer = torch.zeros(self.num_envs, device=self.device)
        self.ee_start_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_start_cart = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_cart = torch.zeros(self.num_envs, 3, device=self.device)
        self.centers_fixed = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_yaw_quat_fixed = env.base_yaw_quat.clone()
        self.ee_yaw_quat_fixed = env.ee_orn.clone()

        self.ee_goal_orn_euler = torch.zeros(self.num_envs, 3, device=self.device)
        # self.ee_goal_orn_euler[:, 0] = np.pi / 2
        self.ee_goal_orn_euler[:, 0] = 0.0
        self.ee_goal_orn_quat = quat_from_euler_xyz(
            self.ee_goal_orn_euler[:, 0], self.ee_goal_orn_euler[:, 1], self.ee_goal_orn_euler[:, 2],
        )
        self.ee_goal_orn_delta_rpy = torch.zeros(self.num_envs, 3, device=self.device)

        self.curr_ee_goal_cart = torch.zeros(self.num_envs, 3, device=self.device)
        self.curr_ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device)

        self.init_start_ee_sphere = torch.tensor(
            self.cfg.goal_ee.ranges.init_pos_start, device=self.device,
        ).unsqueeze(0)
        self.init_end_ee_sphere = torch.tensor(
            self.cfg.goal_ee.ranges.init_pos_end, device=self.device,
        ).unsqueeze(0)

        self.reset_tele_op_flag = False

        # Per-env: when True, base_sample snaps to final goal (no interpolation).
        self.base_sample_far = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Per-env: non-far goals that end at the default-joint EE pose.
        self.base_sample_default = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # EE cart / orn in arm-base frame corresponding to default_joint_angles / arm home.
        self.default_ee_cart_arm_base = torch.tensor(
            getattr(env.cfg.arm, "default_ee_cart_arm_base", env.cfg.arm.init_target_ee_base),
            device=self.device,
            dtype=torch.float,
        )
        default_orn_rpy = getattr(env.cfg.arm, "default_ee_orn_arm_base", [0.0, 0.0, 0.0])
        self.default_ee_orn_arm_base = torch.tensor(
            default_orn_rpy, device=self.device, dtype=torch.float,
        )
        self.default_ee_orn_quat_arm_base = quat_from_euler_xyz(
            self.default_ee_orn_arm_base[0:1],
            self.default_ee_orn_arm_base[1:2],
            self.default_ee_orn_arm_base[2:3],
        ).squeeze(0)
        self._default_ee_cart_ready = True

        if self.cfg.experiment.fine_sample:
            self.sample_category = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.collision_lower_limits = torch.tensor(
            self.cfg.goal_ee.collision_lower_limits, device=self.device, dtype=torch.float,
        )
        self.collision_upper_limits = torch.tensor(
            self.cfg.goal_ee.collision_upper_limits, device=self.device, dtype=torch.float,
        )
        self.underground_limit = self.cfg.goal_ee.underground_limit
        self.num_collision_check_samples = self.cfg.goal_ee.num_collision_check_samples
        self.collision_check_t = torch.linspace(
            0, 1, self.num_collision_check_samples, device=self.device,
        )[None, None, :]
        assert self.cfg.goal_ee.command_mode in ['cart', 'sphere']
        self.sphere_error_scale = torch.tensor(self.cfg.goal_ee.sphere_error_scale, device=self.device)
        self.orn_error_scale = torch.tensor(self.cfg.goal_ee.orn_error_scale, device=self.device)
        self.ee_goal_center_offset = torch.tensor([
            self.cfg.goal_ee.sphere_center.x_offset,
            self.cfg.goal_ee.sphere_center.y_offset,
            self.cfg.goal_ee.sphere_center.z_invariant_offset,
        ], device=self.device).repeat(self.num_envs, 1)

        self.curr_ee_goal_cart_world = self.get_spherical_center() + quat_apply(
            env.base_yaw_quat, self.curr_ee_goal_cart,
        )

    def get_spherical_center(self):
        center = torch.cat([
            self.env.root_states[:, :2],
            torch.zeros(self.num_envs, 1, device=self.device),
        ], dim=1)
        center = center + quat_apply(self.env.base_yaw_quat, self.ee_goal_center_offset)
        return center

    def get_spherical_center_fixed(self, env_ids):
        center = torch.cat([
            self.env.root_states[env_ids, :2].clone(),
            torch.zeros(env_ids.size()[0], 1, device=self.device),
        ], dim=1)
        center = center + quat_apply(
            self.env.base_yaw_quat[env_ids, :], self.ee_goal_center_offset[env_ids, :],
        )
        return center.clone()

    def get_ee_pos_fixed(self, env_ids):
        return self.env.ee_pos[env_ids, :].clone()

    @property
    def use_base_sample(self):
        """Live arm-base frame sampling; only valid when fix_sample is off."""
        return bool(getattr(self.cfg.experiment, "base_sample", False)) and not self.cfg.experiment.fix_sample

    def get_arm_base_pos(self, env_ids=None):
        env = self.env
        if env_ids is None:
            return env.base_pos + quat_apply(env.base_quat, env.arm_base_offset)
        return env.base_pos[env_ids] + quat_apply(
            env.base_quat[env_ids], env.arm_base_offset[env_ids],
        )

    def resample_on_reset(self, env_ids, is_init=False):
        if self.use_base_sample:
            self.base_resample(env_ids, is_init=is_init)
        elif self.cfg.experiment.fine_sample:
            self.fine_resample(env_ids, is_init=is_init)
        else:
            self.resample(env_ids, is_init=is_init)

    def _resample_sphere_once(self, env_ids):
        ranges = self.env.goal_ee_ranges
        self._resample_sphere_with_ranges(
            env_ids, ranges["pos_l"], ranges["pos_p"], ranges["pos_y"],
        )

    def _resample_sphere_with_ranges(self, env_ids, pos_l, pos_p, pos_y):
        self.ee_goal_sphere[env_ids, 0] = torch_rand_float(
            pos_l[0], pos_l[1], (len(env_ids), 1), device=self.device,
        ).squeeze(1)
        self.ee_goal_sphere[env_ids, 1] = torch_rand_float(
            pos_p[0], pos_p[1], (len(env_ids), 1), device=self.device,
        ).squeeze(1)
        self.ee_goal_sphere[env_ids, 2] = torch_rand_float(
            pos_y[0], pos_y[1], (len(env_ids), 1), device=self.device,
        ).squeeze(1)

    def _resample_back_sphere_once(self, env_ids):
        ranges = self.env.goal_ee_ranges
        self.ee_goal_sphere[env_ids, 0] = torch_rand_float(
            ranges["back_pos_l"][0], ranges["back_pos_l"][1], (len(env_ids), 1), device=self.device,
        ).squeeze(1)
        self.ee_goal_sphere[env_ids, 1] = torch_rand_float(
            ranges["pos_p"][0], ranges["pos_p"][1], (len(env_ids), 1), device=self.device,
        ).squeeze(1)
        self.ee_goal_sphere[env_ids, 2] = torch_rand_float(
            ranges["pos_y"][0], ranges["pos_y"][1], (len(env_ids), 1), device=self.device,
        ).squeeze(1)

    def _resample_orn_once(self, env_ids):
        ranges = self.env.goal_ee_ranges
        ee_goal_delta_orn_r = torch_rand_float(
            ranges["delta_orn_r"][0], ranges["delta_orn_r"][1], (len(env_ids), 1), device=self.device,
        )
        ee_goal_delta_orn_p = torch_rand_float(
            ranges["delta_orn_p"][0], ranges["delta_orn_p"][1], (len(env_ids), 1), device=self.device,
        )
        ee_goal_delta_orn_y = torch_rand_float(
            ranges["delta_orn_y"][0], ranges["delta_orn_y"][1], (len(env_ids), 1), device=self.device,
        )
        self.ee_goal_orn_delta_rpy[env_ids, :] = torch.cat(
            [ee_goal_delta_orn_r, ee_goal_delta_orn_p, ee_goal_delta_orn_y], dim=-1,
        )

    def base_resample(self, env_ids, is_init=False):
        """SPHERE-style goals in the live Z1 arm-base frame (base_quat @ arm_base_pos).

        Default-arm subset: interpolate cart current→default_ee_cart_arm_base;
        orientation tracks default_ee_orn_arm_base (arm-base rpy → world via base_quat).
        """
        if len(env_ids) == 0:
            return

        far_ratio = float(getattr(self.cfg.experiment, "base_sample_far_ratio", 0.3))
        self.base_sample_far[env_ids] = torch.rand(len(env_ids), device=self.device) < far_ratio
        far_ids = env_ids[self.base_sample_far[env_ids]]
        near_ids = env_ids[~self.base_sample_far[env_ids]]

        self.base_sample_default[env_ids] = False
        default_ratio = float(getattr(self.cfg.experiment, "base_sample_default_ratio", 0.0))
        if len(near_ids) > 0 and default_ratio > 0:
            default_mask = torch.rand(len(near_ids), device=self.device) < default_ratio
            default_ids = near_ids[default_mask]
            sphere_near_ids = near_ids[~default_mask]
            self.base_sample_default[default_ids] = True
        else:
            default_ids = near_ids[:0]
            sphere_near_ids = near_ids

        self._resample_orn_once(env_ids)
        ranges = self.env.goal_ee_ranges
        if len(sphere_near_ids) > 0:
            self._resample_sphere_with_ranges(
                sphere_near_ids, ranges["pos_l"], ranges["pos_p"], ranges["pos_y"],
            )
        if len(far_ids) > 0:
            self._resample_sphere_with_ranges(
                far_ids,
                ranges.get("base_far_pos_l", ranges["pos_l"]),
                ranges.get("base_far_pos_p", ranges["pos_p"]),
                ranges.get("base_far_pos_y", ranges["pos_y"]),
            )
        if len(sphere_near_ids) > 0 or len(far_ids) > 0:
            non_default = env_ids[~self.base_sample_default[env_ids]]
            if len(non_default) > 0:
                self.ee_goal_cart[non_default, :] = sphere2cart(self.ee_goal_sphere[non_default, :])

        # Default-arm goals: end at configured default EE cart; start = current EE (below).
        if len(default_ids) > 0:
            self.ee_goal_cart[default_ids, :] = self.default_ee_cart_arm_base
            self.ee_goal_sphere[default_ids, :] = cart2sphere(self.ee_goal_cart[default_ids, :])
            self.ee_goal_orn_delta_rpy[default_ids, :] = 0.0

        arm_base_pos = self.get_arm_base_pos(env_ids)
        # Height clip for non-default goals only.
        non_default_ids = env_ids[~self.base_sample_default[env_ids]]
        if len(non_default_ids) > 0:
            z_lo, z_hi = getattr(self.cfg.experiment, "base_sample_z_range", [0.3, 0.8])
            nd_arm = self.get_arm_base_pos(non_default_ids)
            goal_world = nd_arm + quat_apply(
                self.env.base_quat[non_default_ids, :], self.ee_goal_cart[non_default_ids, :],
            )
            goal_world[:, 2] = torch.clip(goal_world[:, 2], float(z_lo), float(z_hi))
            self.ee_goal_cart[non_default_ids, :] = quat_rotate_inverse(
                self.env.base_quat[non_default_ids, :], goal_world - nd_arm,
            )
            self.ee_goal_sphere[non_default_ids, :] = cart2sphere(self.ee_goal_cart[non_default_ids, :])

        # Start = current EE in arm-base frame.
        relative_pos_world = self.env.ee_pos[env_ids, :] - arm_base_pos
        pos_in_base = quat_rotate_inverse(self.env.base_quat[env_ids, :], relative_pos_world)
        self.ee_start_cart[env_ids, :] = pos_in_base
        self.ee_start_sphere[env_ids, :] = cart2sphere(pos_in_base)

        self.goal_timer[env_ids] = 0.0

    def resample(self, env_ids, is_init=False):
        if self.cfg.experiment.tele_op:
            self.centers_fixed[env_ids, :] = self.get_ee_pos_fixed(env_ids)
            ee_yaw = euler_from_quat(self.env.ee_orn)[2]
            self.ee_yaw_quat_fixed = quat_from_euler_xyz(torch.tensor(0), torch.tensor(0), ee_yaw)
            self.reset_tele_op_flag = False
            return

        if len(env_ids) > 0:
            init_env_ids = env_ids.clone()

            if is_init:
                if self.cfg.experiment.fix_sample:
                    self.ee_start_sphere[env_ids] = self.init_start_ee_sphere[:].clone()
                    for _ in range(10):
                        self._resample_sphere_once(env_ids)
                        collision_mask = self._collision_check(env_ids)
                        env_ids = env_ids[collision_mask]
                        if len(env_ids) == 0:
                            break
                else:
                    self.ee_goal_orn_delta_rpy[env_ids, :] = 0
                    self.ee_start_sphere[env_ids] = self.init_start_ee_sphere[:]
                    self.ee_goal_sphere[env_ids] = self.init_end_ee_sphere[:]
            else:
                if not self.cfg.debug.test_vel:
                    self._resample_orn_once(env_ids)
                self.ee_start_sphere[env_ids] = self.ee_goal_sphere[env_ids].clone()
                if not self.cfg.debug.test_vel:
                    for _ in range(10):
                        self._resample_sphere_once(env_ids)
                        collision_mask = self._collision_check(env_ids)
                        env_ids = env_ids[collision_mask]
                        if len(env_ids) == 0:
                            break
            self.ee_goal_cart[init_env_ids, :] = sphere2cart(self.ee_goal_sphere[init_env_ids, :])
            if self.cfg.experiment.fix_sample:
                self.centers_fixed[init_env_ids, :] = self.get_spherical_center_fixed(init_env_ids)
                self.base_yaw_quat_fixed[init_env_ids, :] = self.env.base_yaw_quat[init_env_ids, :].clone()
                relative_pos_world = (
                    self.env.ee_pos[init_env_ids, :].clone() - self.centers_fixed[init_env_ids, :].clone()
                )
                pos_in_fixed = quat_rotate_inverse(self.base_yaw_quat_fixed[init_env_ids, :], relative_pos_world)
                self.ee_start_sphere[init_env_ids] = cart2sphere(pos_in_fixed)

            self.goal_timer[init_env_ids] = 0.0

    def assign_sample_categories(self, env_ids):
        ratios = getattr(self.cfg.experiment, "fine_sample_ratios", [0.4, 0.3, 0.3])
        r_sphere, r_back, r_rot = [float(r) for r in ratios]
        total = r_sphere + r_back + r_rot
        if total <= 0:
            raise ValueError(f"fine_sample_ratios must be positive, got {ratios}")
        t_sphere = r_sphere / total
        t_back = t_sphere + r_back / total

        rand_ids = torch.rand(len(env_ids), device=self.device)
        categories = torch.where(
            rand_ids < t_sphere,
            torch.full((len(env_ids),), self.SAMPLE_SPHERE, dtype=torch.long, device=self.device),
            torch.where(
                rand_ids < t_back,
                torch.full((len(env_ids),), self.SAMPLE_BACK, dtype=torch.long, device=self.device),
                torch.full((len(env_ids),), self.SAMPLE_ROT, dtype=torch.long, device=self.device),
            ),
        )
        self.sample_category[env_ids] = categories

    def _ids_for_category(self, env_ids, category):
        if len(env_ids) == 0:
            return env_ids
        return env_ids[self.sample_category[env_ids] == category]

    def fine_resample(self, env_ids, is_init=False):
        if len(env_ids) > 0:
            self.assign_sample_categories(env_ids)
            sphere_sample_ids = self._ids_for_category(env_ids, self.SAMPLE_SPHERE)
            back_sample_ids = self._ids_for_category(env_ids, self.SAMPLE_BACK)
            rot_sample_ids = self._ids_for_category(env_ids, self.SAMPLE_ROT)

            if len(sphere_sample_ids) > 0:
                self._resample_orn_once(sphere_sample_ids)
                self._resample_sphere_once(sphere_sample_ids)
                self.ee_goal_cart[sphere_sample_ids, :] = sphere2cart(self.ee_goal_sphere[sphere_sample_ids, :])
                self.centers_fixed[sphere_sample_ids, :] = self.get_spherical_center_fixed(sphere_sample_ids)
                self.base_yaw_quat_fixed[sphere_sample_ids, :] = self.env.base_yaw_quat[sphere_sample_ids, :].clone()
                relative_pos_world = (
                    self.env.ee_pos[sphere_sample_ids, :].clone() - self.centers_fixed[sphere_sample_ids, :].clone()
                )
                pos_in_fixed = quat_rotate_inverse(self.base_yaw_quat_fixed[sphere_sample_ids, :], relative_pos_world)
                self.ee_start_sphere[sphere_sample_ids] = cart2sphere(pos_in_fixed)

            if len(back_sample_ids) > 0:
                self._resample_orn_once(back_sample_ids)
                self._resample_back_sphere_once(back_sample_ids)
                self.ee_goal_cart[back_sample_ids, :] = sphere2cart(self.ee_goal_sphere[back_sample_ids, :])
                self.ee_goal_cart[back_sample_ids, 2] = torch.clip(self.ee_goal_cart[back_sample_ids, 2], 0.4, 0.6)
                self.ee_goal_sphere[back_sample_ids, :] = cart2sphere(self.ee_goal_cart[back_sample_ids, :])
                self.centers_fixed[back_sample_ids, :] = self.get_spherical_center_fixed(back_sample_ids)
                self.base_yaw_quat_fixed[back_sample_ids, :] = self.env.base_yaw_quat[back_sample_ids, :].clone()
                relative_pos_world = (
                    self.env.ee_pos[back_sample_ids, :].clone() - self.centers_fixed[back_sample_ids, :].clone()
                )
                pos_in_fixed = quat_rotate_inverse(self.base_yaw_quat_fixed[back_sample_ids, :], relative_pos_world)
                self.ee_start_cart[back_sample_ids, :] = pos_in_fixed
                self.ee_start_sphere[back_sample_ids, :] = cart2sphere(pos_in_fixed)

            if len(rot_sample_ids) > 0:
                self._resample_orn_once(rot_sample_ids)
                self.ee_goal_orn_delta_rpy[rot_sample_ids, :2] = torch.clip(
                    self.ee_goal_orn_delta_rpy[rot_sample_ids, :2], -0.02, 0.02,
                )

                rand_rot_mask = torch.rand(len(rot_sample_ids), device=self.device)
                self.ee_goal_orn_delta_rpy[rot_sample_ids, 2] = torch.where(
                    rand_rot_mask > 0.5,
                    self.ee_goal_orn_delta_rpy[rot_sample_ids, 2] - np.pi / 2,
                    self.ee_goal_orn_delta_rpy[rot_sample_ids, 2] + np.pi / 2,
                )

                n_rot = len(rot_sample_ids)
                self.ee_goal_cart[rot_sample_ids, 0] = 0.1 * torch.rand(n_rot, device=self.device) + 0.5
                self.ee_goal_cart[rot_sample_ids, 1] = 0.1 * torch.rand(n_rot, device=self.device) - 0.05
                self.ee_goal_cart[rot_sample_ids, 2] = 0.1 * torch.rand(n_rot, device=self.device) + 0.5
                self.ee_goal_sphere[rot_sample_ids, :] = cart2sphere(self.ee_goal_cart[rot_sample_ids, :])
                self.centers_fixed[rot_sample_ids, :] = self.get_spherical_center_fixed(rot_sample_ids)
                self.base_yaw_quat_fixed[rot_sample_ids, :] = self.env.base_yaw_quat[rot_sample_ids, :].clone()
                relative_pos_world = (
                    self.env.ee_pos[rot_sample_ids, :].clone() - self.centers_fixed[rot_sample_ids, :].clone()
                )
                pos_in_fixed = quat_rotate_inverse(self.base_yaw_quat_fixed[rot_sample_ids, :], relative_pos_world)
                self.ee_start_cart[rot_sample_ids, :] = pos_in_fixed

            self.goal_timer[env_ids] = 0.0

    def _collision_check(self, env_ids):
        ee_target_all_sphere = torch.lerp(
            self.ee_start_sphere[env_ids, ..., None],
            self.ee_goal_sphere[env_ids, ..., None],
            self.collision_check_t,
        ).squeeze(-1)
        ee_target_cart = sphere2cart(
            torch.permute(ee_target_all_sphere, (2, 0, 1)).reshape(-1, 3),
        ).reshape(self.num_collision_check_samples, -1, 3)
        collision_mask = torch.any(torch.logical_and(
            torch.all(ee_target_cart < self.collision_upper_limits, dim=-1),
            torch.all(ee_target_cart > self.collision_lower_limits, dim=-1),
        ), dim=0)
        underground_mask = torch.any(ee_target_cart[..., 2] < self.underground_limit, dim=0)
        return collision_mask | underground_mask

    def update_curr(self):
        if not self.cfg.env.teleop_mode:
            t = torch.clip(self.goal_timer / self.traj_timesteps, 0, 1)
            if self.use_base_sample:
                # Near: interpolate in sphere; far: snap; default-joint: cart lerp current→home.
                self.curr_ee_goal_sphere[:] = torch.lerp(
                    self.ee_start_sphere, self.ee_goal_sphere, t[:, None],
                )
                if self.base_sample_far.any():
                    self.curr_ee_goal_sphere[self.base_sample_far, :] = self.ee_goal_sphere[self.base_sample_far, :]
                self.curr_ee_goal_cart[:] = sphere2cart(self.curr_ee_goal_sphere)
                if self.base_sample_default.any():
                    self.curr_ee_goal_cart[self.base_sample_default, :] = torch.lerp(
                        self.ee_start_cart[self.base_sample_default, :],
                        self.ee_goal_cart[self.base_sample_default, :],
                        t[self.base_sample_default, None],
                    )
                    self.curr_ee_goal_sphere[self.base_sample_default, :] = cart2sphere(
                        self.curr_ee_goal_cart[self.base_sample_default, :],
                    )
            elif self.cfg.experiment.fine_sample:
                sphere_mask = self.sample_category == self.SAMPLE_SPHERE
                back_mask = self.sample_category == self.SAMPLE_BACK
                rot_mask = self.sample_category == self.SAMPLE_ROT
                if sphere_mask.any():
                    self.curr_ee_goal_sphere[sphere_mask, :] = torch.lerp(
                        self.ee_start_sphere[sphere_mask, :],
                        self.ee_goal_sphere[sphere_mask, :],
                        t[sphere_mask, None],
                    )
                if back_mask.any():
                    self.curr_ee_goal_cart[back_mask, :] = torch.lerp(
                        self.ee_start_cart[back_mask, :],
                        self.ee_goal_cart[back_mask, :],
                        t[back_mask, None],
                    )
                if rot_mask.any():
                    self.curr_ee_goal_cart[rot_mask, :] = torch.lerp(
                        self.ee_start_cart[rot_mask, :],
                        self.ee_goal_cart[rot_mask, :],
                        t[rot_mask, None],
                    )
            else:
                self.curr_ee_goal_sphere[:] = torch.lerp(self.ee_start_sphere, self.ee_goal_sphere, t[:, None])

        default_yaw = torch.zeros(self.num_envs, device=self.device)

        if self.use_base_sample:
            # curr_ee_goal_cart already set above; world goal tracks live arm base + base_quat.
            pass
        else:
            # With far_sample: radius>1 snaps to final goal; radius<=1 keeps interpolation.
            is_far = self.ee_goal_sphere[:, 0] > 1.0

            if self.cfg.experiment.fine_sample and self.cfg.experiment.far_sample:
                sphere_mask = self.sample_category == self.SAMPLE_SPHERE
                back_mask = self.sample_category == self.SAMPLE_BACK
                rot_mask = self.sample_category == self.SAMPLE_ROT

                sphere_far = sphere_mask & is_far
                sphere_near = sphere_mask & ~is_far
                if sphere_far.any():
                    self.curr_ee_goal_cart[sphere_far, :] = sphere2cart(self.ee_goal_sphere[sphere_far, :])
                if sphere_near.any():
                    self.curr_ee_goal_cart[sphere_near, :] = sphere2cart(self.curr_ee_goal_sphere[sphere_near, :])

                back_far = back_mask & is_far
                if back_far.any():
                    self.curr_ee_goal_cart[back_far, :] = self.ee_goal_cart[back_far, :]

                rot_far = rot_mask & is_far
                if rot_far.any():
                    self.curr_ee_goal_cart[rot_far, :] = self.ee_goal_cart[rot_far, :]
            elif self.cfg.experiment.fine_sample and not self.cfg.experiment.far_sample:
                sphere_mask = self.sample_category == self.SAMPLE_SPHERE
                if sphere_mask.any():
                    self.curr_ee_goal_cart[sphere_mask, :] = sphere2cart(self.curr_ee_goal_sphere[sphere_mask, :])
            elif self.cfg.experiment.far_sample:
                if is_far.any():
                    self.curr_ee_goal_cart[is_far, :] = sphere2cart(self.ee_goal_sphere[is_far, :])
                if (~is_far).any():
                    self.curr_ee_goal_cart[~is_far, :] = sphere2cart(self.curr_ee_goal_sphere[~is_far, :])
            else:
                self.curr_ee_goal_cart[:] = sphere2cart(self.curr_ee_goal_sphere)

        if self.use_base_sample:
            arm_base_pos = self.get_arm_base_pos()
            ee_goal_cart_base_global = quat_apply(self.env.base_quat, self.curr_ee_goal_cart)
            default_yaw[:] = torch.atan2(ee_goal_cart_base_global[:, 1], ee_goal_cart_base_global[:, 0])
            # Default-arm EE goals: no reach-aligned yaw; orn set from default_ee_orn_arm_base below.
            if self.base_sample_default.any():
                default_yaw[self.base_sample_default] = 0.0
            self.curr_ee_goal_cart_world = arm_base_pos + ee_goal_cart_base_global
            z_lo, z_hi = getattr(self.cfg.experiment, "base_sample_z_range", [0.3, 0.8])
            non_default = ~self.base_sample_default
            if non_default.any():
                self.curr_ee_goal_cart_world[non_default, 2] = torch.clip(
                    self.curr_ee_goal_cart_world[non_default, 2], float(z_lo), float(z_hi),
                )
                # Keep arm-base cart consistent with the clipped world goal (obs/IK/sphere pitch).
                self.curr_ee_goal_cart[non_default, :] = quat_rotate_inverse(
                    self.env.base_quat[non_default],
                    self.curr_ee_goal_cart_world[non_default] - arm_base_pos[non_default],
                )
                self.curr_ee_goal_sphere[non_default, :] = cart2sphere(
                    self.curr_ee_goal_cart[non_default, :],
                )
        elif self.cfg.experiment.fix_sample and not self.cfg.experiment.fine_sample:
            ee_goal_cart_yaw_global = quat_apply(self.base_yaw_quat_fixed, self.curr_ee_goal_cart)
            self.curr_ee_goal_cart_world = self.centers_fixed + ee_goal_cart_yaw_global
            self.curr_ee_goal_cart_world[:, 2] = torch.clip(self.curr_ee_goal_cart_world[:, 2], 0.3, 0.8)
        elif self.cfg.experiment.fix_sample and self.cfg.experiment.fine_sample:
            for category in (self.SAMPLE_SPHERE, self.SAMPLE_BACK, self.SAMPLE_ROT):
                mask = self.sample_category == category
                if not mask.any():
                    continue
                ee_goal_cart_yaw_global = quat_apply(
                    self.base_yaw_quat_fixed[mask, :], self.curr_ee_goal_cart[mask, :],
                )
                default_yaw[mask] = torch.atan2(ee_goal_cart_yaw_global[:, 1], ee_goal_cart_yaw_global[:, 0])
                self.curr_ee_goal_cart_world[mask, :] = self.centers_fixed[mask, :] + ee_goal_cart_yaw_global
            self.curr_ee_goal_cart_world[:, 2] = torch.clip(self.curr_ee_goal_cart_world[:, 2], 0.3, 0.8)
        elif self.cfg.experiment.tele_op:
            ee_goal_cart_yaw_global = quat_apply(self.base_yaw_quat_fixed, self.curr_ee_goal_cart)
        else:
            ee_goal_cart_yaw_global = quat_apply(self.env.base_yaw_quat, self.curr_ee_goal_cart)
            self.curr_ee_goal_cart_world = self.get_spherical_center() + ee_goal_cart_yaw_global

        if self.cfg.experiment.use_rfm:
            # default_pitch = -self.ee_goal_sphere[:, 1] + self.cfg.goal_ee.arm_induced_pitch
            default_pitch = -self.ee_goal_sphere[:, 1]
        else:
            # default_pitch = -self.curr_ee_goal_sphere[:, 1] + self.cfg.goal_ee.arm_induced_pitch
            default_pitch = -self.curr_ee_goal_sphere[:, 1]
        if self.use_base_sample and self.base_sample_default.any():
            default_pitch = default_pitch.clone()
            default_pitch[self.base_sample_default] = 0.0
            self.ee_goal_orn_delta_rpy[self.base_sample_default, :] = 0.0
        self.ee_goal_orn_quat = quat_from_euler_xyz(
            # self.ee_goal_orn_delta_rpy[:, 0] + np.pi / 2,
            self.ee_goal_orn_delta_rpy[:, 0],
            self.ee_goal_orn_delta_rpy[:, 1] + default_pitch,
            # self.ee_goal_orn_delta_rpy[:, 1],
            self.ee_goal_orn_delta_rpy[:, 2] + default_yaw,
        )
        if self.use_base_sample and self.base_sample_default.any():
            mask = self.base_sample_default
            n = int(mask.sum().item())
            orn_local = self.default_ee_orn_quat_arm_base.unsqueeze(0).expand(n, -1)
            self.ee_goal_orn_quat[mask] = quat_mul(self.env.base_quat[mask], orn_local)

        self.goal_timer += 1
        resample_id = (self.goal_timer > self.traj_total_timesteps).nonzero(as_tuple=False).flatten()

        if len(resample_id) > 0 and self.env.stop_update_goal:
            self.env.commands[resample_id, 0] = 0
            self.env.commands[resample_id, 1] = 0
            self.env.commands[resample_id, 2] = 0
        if self.use_base_sample:
            self.base_resample(resample_id, is_init=False)
        elif self.cfg.experiment.fine_sample:
            self.fine_resample(resample_id, is_init=False)
        else:
            self.resample(resample_id)

    def reset_goal_timer(self, env_ids):
        self.goal_timer[env_ids] = 0.
