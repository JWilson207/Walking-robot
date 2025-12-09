# SPDX-License-Identifier: BSD-3-Clause
# MuJoCo adaptation of Isaac Gym base task

import sys
import numpy as np
import torch
import mujoco
from typing import Optional

class BaseTask:
    """Base class for RL tasks using MuJoCo simulator - Headless only for server training"""

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        self.headless = True  # Force headless mode for server
        
        # Determine device
        if sim_device == 'cuda' or sim_device.startswith('cuda:'):
            self.device = sim_device
        else:
            self.device = 'cpu'
        
        self.num_envs = cfg.env.num_envs
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions
        
        # Optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        
        # Allocate buffers
        self.obs_buf = torch.zeros(
            self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.neg_reward_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.pos_reward_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool)
        self.random_half_phase = torch.pi * torch.randint(0, 2,
            (1, self.num_envs), device=self.device, dtype=torch.long)
        
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(
                self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float)
        else:
            self.privileged_obs_buf = None
        
        self.extras = {}
        
        # Create simulation
        #self.create_sim()
        self.enable_viewer_sync = False  # No viewer in headless mode
        self.viewer = None
    
    def get_observations(self):
        return self.obs_buf
    
    def get_privileged_observations(self):
        return self.privileged_obs_buf
    
    def get_rma_observations(self):
        return self.rma_obs_buf
    
    def reset(self):
        """Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(
            self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs
    
    def step(self, actions):
        raise NotImplementedError
    
    def render(self, sync_frame_time=True):
        """Render is disabled in headless mode for server training"""
        pass
    
    def close(self):
        """Clean up resources"""
        pass
    
    def __del__(self):
        self.close()
