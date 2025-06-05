#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''

Mass Spring Damper Environment

Author: Deniz Canbay


'''

'''
Tasks

1) Switch to PyGame for rendering


'''
import importlib
import numpy as np
import gymnasium as gym  # Changed from gym
from gymnasium import spaces  # Changed from gym.spaces
from gymnasium.utils import seeding  # Changed from gym.utils
from os import path


class MassEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, m=1, k=10, c=0.1):
        self.m = m  # mass
        self.k = k  # spring constant
        self.c = c  # damping coefficient
        self.dt = 0.02
        
        # State is [position, velocity]
        high = np.array([10.0, 10.0], dtype=np.float32)
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        
        self.state = None

    def step(self, action):
        x, x_dot = self.state
        force = action[0]
        
        # Mass-spring-damper dynamics: m*x_ddot + c*x_dot + k*x = F
        x_ddot = (force - self.c * x_dot - self.k * x) / self.m
        
        # Euler integration
        x_new = x + self.dt * x_dot
        x_dot_new = x_dot + self.dt * x_ddot
        
        self.state = np.array([x_new, x_dot_new])
        
        # Reward is negative distance from origin
        reward = -(x_new**2 + 0.1*x_dot_new**2 + 0.001*force**2)
        
        return self.state.copy(), reward, False, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(low=-1, high=1, size=(2,))
        return self.state.copy(), {}

    def render(self):
        pass

    def close(self):
        pass

