# SPDX-License-Identifier: BSD-3-Clause
# MuJoCo adaptation of Pai environment

from humanoid.utils.mujoco_torch_utils import *
from humanoid.envs.base.legged_robot import LeggedRobot, get_euler_xyz_tensor
import torch
import numpy as np


class PaiFreeEnv(LeggedRobot):
    """
    PaiFreeEnv is a class that represents a custom environment for a legged robot using MuJoCo.
    """

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.last_feet_z = 0.05
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.reset()
        self.compute_observations()
    
    def _push_robots(self):
        """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        max_push_angular = self.cfg.domain_rand.max_push_ang_vel
        self.rand_push_force[:, :2] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device
        )  # lin vel x/y
        self.root_states[:, 7:9] = self.rand_push_force[:, :2]

        self.rand_push_torque = torch_rand_float(
            -max_push_angular, max_push_angular, (self.num_envs, 3), device=self.device
        )

        self.root_states[:, 10:13] = self.rand_push_torque

        # Apply to MuJoCo simulation
        for env_id in range(self.num_envs):
            self.data[env_id].qvel[:2] = self.root_states[env_id, 7:9].cpu().numpy()
            self.data[env_id].qvel[3:6] = self.root_states[env_id, 10:13].cpu().numpy()

    def check_termination(self):
        """Check if environments need to be reset"""
        self.reset_buf = torch.any(
            torch.norm(
                self.contact_forces[:, self.termination_contact_indices, :], dim=-1
            )
            > 1.0,
            dim=1,
        )
        self.time_out_buf = (
            self.episode_length_buf > self.max_episode_length
        )  # no terminal reward for time-outs
        self.reset_buf |= torch.any(
            torch.abs(self.projected_gravity[:, 0:1]) > 0.8, dim=1
        )
        self.reset_buf |= torch.any(
            torch.abs(self.projected_gravity[:, 1:2]) > 0.8, dim=1
        )

        self.reset_buf |= torch.any(self.base_pos[:, 2:3] < 0.3, dim=1)
        self.reset_buf |= self.time_out_buf

    def _get_phase(self):
        cycle_time = self.cfg.rewards.cycle_time
        phase = self.episode_length_buf * self.dt / cycle_time
        return phase

    def _get_gait_phase(self):
        # return float mask 1 is stance, 0 is swing
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase + self.random_half_phase[0])
        # Add double support phase
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        # left foot stance
        stance_mask[:, 0] = sin_pos >= 0
        # right foot stance
        stance_mask[:, 1] = sin_pos < 0
        # Double support phase
        stance_mask[torch.abs(sin_pos) < 0.1] = 1

        return stance_mask

    def compute_ref_state(self):
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase + self.random_half_phase[0])
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        scale_1 = self.cfg.rewards.target_joint_pos_scale
        scale_2 = 2 * scale_1
        # left foot stance phase set to default joint pos
        sin_pos_l[sin_pos_l > 0] = 0
        self.ref_dof_pos[:, 0] = sin_pos_l * scale_1
        self.ref_dof_pos[:, 3] = -sin_pos_l * scale_2
        self.ref_dof_pos[:, 4] = sin_pos_l * scale_1
        # right foot stance phase set to default joint pos
        sin_pos_r[sin_pos_r < 0] = 0
        self.ref_dof_pos[:, 6] = -sin_pos_r * scale_1
        self.ref_dof_pos[:, 9] = sin_pos_r * scale_2
        self.ref_dof_pos[:, 10] = -sin_pos_r * scale_1
        # Double support phase
        self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0

    def compute_observations(self):
        """Computes observations"""
        self.compute_ref_state()
        
        # Prepare observation components
        phase = self._get_phase().unsqueeze(1)
        sin_pos = torch.sin(2 * torch.pi * phase).squeeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).squeeze(1)
        
        # Build observation
        obs_buf = torch.cat([
            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.projected_gravity,  # 3
            self.commands[:, :3] * self.commands_scale,  # 3
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # num_actions
            self.dof_vel * self.obs_scales.dof_vel,  # num_actions
            self.actions,  # num_actions
            sin_pos.unsqueeze(1),  # 1
            cos_pos.unsqueeze(1),  # 1
        ], dim=-1)
        
        # Add noise if configured
        if self.cfg.noise.add_noise:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.cfg.noise.noise_level
        
        # Expand to match num_observations with frame stacking
        single_obs_size = obs_buf.shape[1]
        
        # Frame stacking
        if not hasattr(self, 'obs_history'):
            self.obs_history = torch.zeros(
                self.num_envs, self.cfg.env.frame_stack, single_obs_size,
                device=self.device, dtype=torch.float)
        
        # Update history
        self.obs_history = torch.roll(self.obs_history, 1, dims=1)
        self.obs_history[:, 0] = obs_buf
        
        # Flatten history for observation buffer
        self.obs_buf = self.obs_history.reshape(self.num_envs, -1)
        
        # Privileged observations
        if self.num_privileged_obs is not None:
            priv_obs = torch.cat([
                obs_buf,
                self.contact_forces[:, self.feet_indices, 2].flatten(1),  # foot contact forces
                self.base_lin_vel,  # 3
                self.base_ang_vel,  # 3
            ], dim=-1)
            
            if not hasattr(self, 'critic_history'):
                self.critic_history = torch.zeros(
                    self.num_envs, self.cfg.env.c_frame_stack, priv_obs.shape[1],
                    device=self.device, dtype=torch.float)
            
            self.critic_history = torch.roll(self.critic_history, 1, dims=1)
            self.critic_history[:, 0] = priv_obs
            self.privileged_obs_buf = self.critic_history.reshape(self.num_envs, -1)

    # ======================== REWARD FUNCTIONS ========================

    def _reward_tracking_lin_vel(self):
        """Tracks linear velocity commands along the xy axes."""
        error = self.commands[:, :2] - self.base_lin_vel[:, :2]
        error *= 1.0 / (1.0 + torch.abs(self.commands[:, :2]))
        rew = self._neg_sqrd_exp(error, a=self.cfg.rewards.tracking_sigma_lin).sum(dim=1) / 2
        return rew

    def _reward_tracking_ang_vel(self):
        """Tracks angular velocity commands for yaw rotation."""
        error = self.commands[:, 2] - self.base_ang_vel[:, 2]
        error *= 1.0 / (1.0 + torch.abs(self.commands[:, 2]))
        rew = self._neg_sqrd_exp(error, a=self.cfg.rewards.tracking_sigma_ang)
        return rew

    def _reward_feet_clearance(self):
        """Calculates reward based on the clearance of the swing leg from the ground."""
        # Compute feet contact mask
        contact = self.contact_forces[:, self.feet_indices, 2] > 0.1

        # Get the z-position of the feet and compute the change in z-position
        feet_z = (self.rigid_state[:, self.feet_indices - 1, 2]) - 0.05063
        delta_z = feet_z - self.last_feet_z
        self.feet_height += delta_z
        self.last_feet_z = feet_z

        # Compute swing mask
        swing_mask = 1 - self._get_gait_phase()

        # feet height should be closed to target feet height at the peak
        rew_pos = (
            torch.abs(self.feet_height - self.cfg.rewards.target_feet_height) < 0.01
        )
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
        self.feet_height *= ~contact
        return rew_pos

    def _reward_low_speed(self):
        """Rewards or penalizes the robot based on its speed relative to the commanded speed."""
        absolute_speed = torch.abs(self.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.commands[:, 0])

        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)

        sign_mismatch = torch.sign(self.base_lin_vel[:, 0]) != torch.sign(
            self.commands[:, 0]
        )

        reward = torch.zeros_like(self.base_lin_vel[:, 0])
        reward[speed_too_low] = -1.0
        reward[speed_too_high] = 0.0
        reward[speed_desired] = 1.2
        reward[sign_mismatch] = -2.0
        return reward * (self.commands[:, 0].abs() > 0.1)

    def _reward_torques(self):
        """Penalizes the use of high torques."""
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        """Penalizes high velocities at the DOF."""
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        """Penalizes high accelerations at the DOF."""
        return torch.sum(
            torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1
        )

    def _reward_collision(self):
        """Penalizes collisions."""
        return torch.sum(
            1.0 * (
                torch.norm(
                    self.contact_forces[:, self.penalised_contact_indices, :], dim=-1
                )
                > 0.1
            ),
            dim=1,
        )

    def _reward_action_smoothness(self):
        """Encourages smoothness in actions."""
        term_1 = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(
            torch.square(self.actions + self.last_last_actions - 2 * self.last_actions),
            dim=1,
        )
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return term_1 + term_2 + term_3

    def _reward_orientation(self):
        """Penalize non-flat base orientation."""
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        """Penalize deviation from target base height."""
        return torch.square(self.base_pos[:, 2] - self.cfg.rewards.base_height_target)

    def _reward_base_acc(self):
        """Penalize base accelerations."""
        return torch.sum(torch.square(self.root_states[:, 7:10] - self.last_root_vel[:, :3]), dim=1)

    def _reward_feet_contact_forces(self):
        """Penalize high contact forces on feet."""
        return torch.sum(
            (torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - 
             self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_default_ankle_roll_pos(self):
        """Reward keeping ankle roll at default position."""
        e_1 = get_euler_xyz_tensor(self.rigid_state[:, self.feet_indices[0], 3:7])
        e_2 = get_euler_xyz_tensor(self.rigid_state[:, self.feet_indices[1], 3:7])

        feet_eular_0 = torch.abs(e_1[:, 0])
        feet_eular_1 = torch.abs(e_2[:, 0])
        rew = torch.exp(-((feet_eular_0 + feet_eular_1) / 2) / 0.1)
        feet_eular_0 = torch.abs(e_1[:, 1])
        feet_eular_1 = torch.abs(e_2[:, 1])
        rew += torch.exp(-((feet_eular_0 + feet_eular_1) / 2) / 0.1)
        return rew / 2

    def _reward_termination(self):
        """Terminal reward / penalty"""
        return -(self.reset_buf * ~self.time_out_buf).float()

    # ======================== HELPER FUNCTIONS ========================

    def _neg_exp(self, x, a=1):
        """Shorthand helper for negative exponential e^(-x/a)"""
        return torch.exp(-(x / a) / a)

    def _neg_sqrd_exp(self, x, a=1):
        """Shorthand helper for negative squared exponential e^(-(x/a)^2)"""
        return torch.exp(-torch.square(x / a) / a)