'''

Pendulum Environment

with continuous actions and state space

Author: Deniz Ekin Canbay

'''

import importlib
import gymnasium as gym  # Changed from gym
import numpy as np
from gymnasium import spaces  # Changed from gym.spaces
from gymnasium.utils import seeding  # Changed from gym.utils
from os import path
import math



class PendulumEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}  # Updated metadata

    def __init__(self, m=1.0, mp=0, b=0.0, l=1.0, max_torque=2.0, random_start=True):
        self.max_speed = 8
        self.max_torque = max_torque
        self.dt = 0.05
        self.g = 9.807
        self.m = m
        self.l = l
        self.b = b
        self.mp = mp
        self.random_start = random_start
        self.viewer = None

        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def step(self, u):
        th, thdot = self.state

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u
        costs = self._angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)

        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])
        
        # Gymnasium API: obs, reward, terminated, truncated, info
        return self._get_obs(), -costs, False, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.random_start:
            high = np.array([np.pi, 1])
            self.state = self.np_random.uniform(low=-high, high=high)
        else:
            self.state = np.array([np.pi, 0])
        self.last_u = None
        
        # Gymnasium API: obs, info
        return self._get_obs(), {}

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

    def render(self):
        # Simplified render for now
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def _angle_normalize(self, x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi


class PendulumDiscEnv(gym.Env):
    def __init__(self, m=1.0, mp=0, b=0.0, l=1.0, max_torque=2.0, random_start=True):
        # Similar implementation but with discrete actions
        pass

