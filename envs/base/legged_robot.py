# SPDX-License-Identifier: BSD-3-Clause
# MuJoCo adaptation of Isaac Gym legged robot

import os
import numpy as np
import torch
import mujoco
import mujoco.viewer
from collections import deque
from typing import Dict, Tuple

from humanoid.utils.mujoco_torch_utils import *
from humanoid.envs.base.base_task import BaseTask
from humanoid.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from humanoid.utils.helpers import class_to_dict


def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_xyz(quat)
    euler_xyz = torch.stack((r, p, w), dim=1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz


class LeggedRobot(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        """Parses the provided config file, creates simulation and environments,
        initializes pytorch buffers used during training.
        
        Args:
            cfg (Dict): Environment config file
            sim_params: simulation parameters
            physics_engine: Physics engine type (ignored for MuJoCo)
            sim_device (string): 'cuda' or 'cpu'
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        
        if not self.headless:
            np.sqrt(diff[0]**2 + diff[1]**2)
    
    def _init_buffers(self):
        """Initialize buffers for tracking robot state"""
        self.base_quat = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.float)
        self.base_pos = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.base_lin_vel = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.base_ang_vel = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.projected_gravity = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.base_euler_xyz = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        
        # Gravity vector
        self.gravity_vec = torch.tensor([0., 0., -1.], device=self.device, 
                                       dtype=torch.float).repeat((self.num_envs, 1))
        
        # Actions and torques
        self.actions = torch.zeros(self.num_envs, self.num_actions, 
                                   device=self.device, dtype=torch.float)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, 
                                       device=self.device, dtype=torch.float)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, 
                                            device=self.device, dtype=torch.float)
        self.torques = torch.zeros(self.num_envs, self.num_actions, 
                                   device=self.device, dtype=torch.float)
        
        # Commands
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, 
                                   device=self.device, dtype=torch.float)
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, 
                                           self.obs_scales.ang_vel], 
                                          device=self.device, dtype=torch.float)
        
        # Tracking
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros(self.num_envs, 6, device=self.device, dtype=torch.float)
        self.last_rigid_state = torch.zeros_like(self.rigid_state)
        
        # Feet tracking
        self.feet_air_time = torch.zeros(self.num_envs, len(self.feet_indices), 
                                        device=self.device, dtype=torch.float)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), 
                                        device=self.device, dtype=torch.bool)
        
        # Step counter
        self.common_step_counter = 0
        
        # Random push forces
        self.rand_push_force = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.rand_push_torque = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        
        # DOF properties
        self._prepare_dof_props()
    
    def _prepare_dof_props(self):
        """Prepare DOF properties from config"""
        self.dof_names = []
        for joint_name in self.cfg.init_state.default_joint_angles.keys():
            self.dof_names.append(joint_name)
        
        # Get stiffness and damping
        self.p_gains = torch.zeros(self.num_actions, device=self.device, dtype=torch.float)
        self.d_gains = torch.zeros(self.num_actions, device=self.device, dtype=torch.float)
        
        for i, joint_name in enumerate(self.dof_names):
            # Find matching pattern in config
            for key in self.cfg.control.stiffness.keys():
                if key in joint_name:
                    self.p_gains[i] = self.cfg.control.stiffness[key]
                    self.d_gains[i] = self.cfg.control.damping[key]
                    break
        
        # Default joint positions
        self.default_dof_pos = torch.zeros(self.num_actions, device=self.device, dtype=torch.float)
        for i, joint_name in enumerate(self.dof_names):
            if joint_name in self.cfg.init_state.default_joint_angles:
                self.default_dof_pos[i] = self.cfg.init_state.default_joint_angles[joint_name]
    
    def _compute_torques(self, actions):
        """Compute torques from actions using PD control"""
        actions_scaled = actions * self.cfg.control.action_scale
        target_pos = actions_scaled + self.default_dof_pos
        
        torques = self.p_gains * (target_pos - self.dof_pos) - self.d_gains * self.dof_vel
        return torch.clip(torques, -80., 80.)  # Safety limit
    
    def _post_physics_step_callback(self):
        """Callback for post-physics step computations"""
        # Update feet air time
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > 1.0
        self.feet_air_time += self.dt
        self.feet_air_time *= ~contact
        
        # Random pushes
        if self.cfg.domain_rand.push_robots and \
           (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
        
        # Resample commands
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
    
    def _push_robots(self):
        """Randomly push robots"""
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        max_ang = self.cfg.domain_rand.max_push_ang_vel
        
        self.rand_push_force[:, :2] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device)
        self.rand_push_torque[:] = torch_rand_float(
            -max_ang, max_ang, (self.num_envs, 3), device=self.device)
        
        # Apply to root states
        self.root_states[:, 7:9] = self.rand_push_force[:, :2]
        self.root_states[:, 10:13] = self.rand_push_torque
        
        # Apply to MuJoCo data
        for env_id in range(self.num_envs):
            self.data[env_id].qvel[:2] = self.rand_push_force[env_id, :2].cpu().numpy()
            self.data[env_id].qvel[3:6] = self.rand_push_torque[env_id].cpu().numpy()
    
    def _resample_commands(self, env_ids):
        """Resample commands for specified environments"""
        if len(env_ids) == 0:
            return
        
        # Sample new commands
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0], 
            self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0],
            self.command_ranges["lin_vel_y"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)
        
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0],
                self.command_ranges["heading"][1],
                (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(
                self.command_ranges["ang_vel_yaw"][0],
                self.command_ranges["ang_vel_yaw"][1],
                (len(env_ids), 1), device=self.device).squeeze(1)
        
        # Set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)
    
    def _prepare_reward_function(self):
        """Prepare reward function and scales"""
        self.reward_functions = {}
        self.reward_names = []
        
        for name, scale in self.reward_scales.items():
            if scale == 0:
                continue
            self.reward_names.append(name)
            self.reward_functions[name] = getattr(self, f'_reward_{name}')
        
        # Initialize reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, device=self.device, 
                                               dtype=torch.float) 
                            for name in self.reward_names}
    
    def compute_reward(self):
        """Compute total reward"""
        self.rew_buf[:] = 0.
        
        for name in self.reward_names:
            rew = self.reward_functions[name]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        
        # Clip negative rewards if configured
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf, min=0.)
        
        # Reset episode sums for done environments
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        for name in self.reward_names:
            self.episode_sums[name][env_ids] = 0.
    
    def compute_observations(self):
        """Compute observations - to be implemented by subclass"""
        raise NotImplementedError
    
    # Reward functions (examples - implement as needed)
    def _reward_tracking_lin_vel(self):
        """Reward tracking linear velocity command"""
        error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-error / self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        """Reward tracking angular velocity command"""
        error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-error / self.cfg.rewards.tracking_sigma)
    
    def _reward_torques(self):
        """Penalize torque usage"""
        return -torch.sum(torch.square(self.torques), dim=1)
    
    def _reward_dof_vel(self):
        """Penalize DOF velocities"""
        return -torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        """Penalize DOF accelerations"""
        return -torch.sum(torch.square((self.dof_vel - self.last_dof_vel) / self.dt), dim=1)
    
    def _reward_collision(self):
        """Penalize collisions"""
        return -torch.sum(
            (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], 
                       dim=-1) > 0.1).float(), dim=1)
    
    def _reward_termination(self):
        """Penalize termination"""
        return -(self.reset_buf * ~self.time_out_buf).float()
    
    def _draw_debug_vis(self):
        """Draw debug visualization"""
        pass  # TODO: Implement if needed
