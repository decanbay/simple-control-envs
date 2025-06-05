import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from os import path


class PendulumEnv2(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, m=1.0, l=1.0, g=9.807, max_torque=2.0, random_start=True):
        self.max_speed = 8
        self.max_torque = max_torque
        self.dt = 0.02
        self.g = g
        self.m = m
        self.l = l
        self.viewer = None
        self.random_start = random_start

        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def step(self, action):
        if action == 0:
            u = -self.max_torque
        elif action == 1:
            u = 0.0
        else:
            u = self.max_torque

        th, thdot = self.state

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        self.last_u = u
        costs = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)

        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.random_start:
            high = np.array([np.pi, 1])
            self.state = self.np_random.uniform(low=-high, high=high)
        else:
            self.state = np.array([np.pi, 0])
        self.last_u = None
        return self._get_obs(), {}

    def _get_obs(self):
        """Return observation in format [cos(theta), sin(theta), theta_dot]"""
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

    def render(self):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi
