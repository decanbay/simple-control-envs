'''

Pendulum Environment

with continuous actions and state space

Author: Deniz Canbay

'''

import importlib
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from os import path



class PendulumEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, m=1.0, mp=0, b=0.0,l=1.0, max_torque=2.0, random_start=True):
        super().__init__()
        self.g = 9.807
        self.dt = 0.05
        self.dt_ode = self.dt
        self.m = m # mass of the rod
        self.mp = mp  # mass attached to tip
        self.L = l
        self.b = b
        self.I0 = (m * l**2)/3 + mp* l**2  #mass moment of inertia
        self.max_torque = max_torque
        self.max_rot_speed = 8
        ALPHA_MAX = 2*np.pi
        obs_high = np.asarray([ALPHA_MAX, np.inf],
                              dtype=np.float32)
        self.random_start = random_start
        self.observation_space = spaces.Box(-obs_high, obs_high,
                                            dtype=np.float32)
        self.action_space = spaces.Box(low=-max_torque,
                                       high=max_torque,
                                       shape=(1,),
                                       dtype=np.float32)
        self.states = []
        self._max_episode_steps = int(10/self.dt)
        self.max_episode_steps=self._max_episode_steps
        self.viewer = None


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    

    def reset(self):
        # self.state = np.concatenate(([0],np.random.uniform(low = np.pi-0.1, high=np.pi+0.1, size=1),[0],np.random.uniform(low = -np.pi, high=np.pi, size=1)))
        if self.random_start:
            self.state = np.array([np.random.uniform(low = np.pi-0.1, high=np.pi+0.1), np.random.uniform(low = -np.pi, high=np.pi)], dtype=np.float32)
        else:
            self.state = np.array([np.pi, 0], dtype=np.float32)
        self.states.append(self.state)
        self.episode_steps = 0
        self.last_u = None
        return self.state

    def reset_random(self):
        # self.state = np.concatenate(([0],np.random.uniform(low = np.pi-0.1, high=np.pi+0.1, size=1),[0],np.random.uniform(low = -np.pi, high=np.pi, size=1)))
        self.state = np.array([np.random.uniform(low =-2*np.pi, high=2*np.pi), np.random.uniform(low = -5*np.pi, high=5*np.pi)], dtype=np.float32)
        self.states.append(self.state)
        self.vis = None
        self.episode_steps = 0
        return self.state

    def reset_up(self):
        self.state = np.array([np.random.uniform(low =-0.1, high=0.1), np.random.uniform(low = -np.pi, high=np.pi)], dtype=np.float32)
        self.states.append(self.state)
        self.vis = None
        self.episode_steps = 0
        return self.state
    
    def calc_reward(self,u):
        alpha, alpha_d = self.state
        alpha2 =self._angle_normalize(alpha)
        # rew = np.cos(alpha) - 0.01*np.abs(alpha_d)
        r = -(alpha2**2 + 0.1*alpha_d**2 + 0.001 * u**2) # From GYM 
        return r

    def step(self, u):
        u = np.clip(u, -self.max_torque, self.max_torque)
        self.last_u = u  # for rendering
        g = self.g
        L = self.L
        m = self.m
        mp = self.mp
        I0 = self.I0
        b = self.b
        alpha, alpha_d = self.state
        alpha = self._angle_normalize(alpha)
        alpha_dd = ((mp+m/2)*g*L*np.sin(alpha) - b*alpha_d + u[0])/I0
        alpha_d_ = alpha_d + alpha_dd * self.dt
        alpha_ = alpha + alpha_d_*self.dt
        alpha_d_ = np.clip(alpha_d_, -self.max_rot_speed, self.max_rot_speed)
        reward = self.calc_reward(u[0])
        self.state = np.array([alpha_, alpha_d_])
        self.states.append(self.state)
        #reward = 1 - abs(alpha)/np.pi- abs(alpha_d_)/(20*np.pi)
        self.episode_steps += 1
        done = self.episode_steps > self._max_episode_steps
        obs_noise = np.array([np.random.randn()/100, np.random.randn()/100], dtype=np.float32)
        obs_noise = 0
        done=False
        return self.state+obs_noise, reward, done, {}
    

    def render(self, mode="human"):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, 0.2)
            rod.set_color(0.8, 0.3, 0.3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(0.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1.0, 1.0)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u is not None:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


    def _angle_normalize(self, x):
        norm_ang = ((x+np.pi) % (2*np.pi)) - np.pi
        return norm_ang



class PendulumDiscEnv(gym.Env):
    def __init__(self, m=1.0, mp=0, b=0.0,l=1.0, max_torque=2.0, random_start=True):
        super().__init__()
        self.g = 9.807
        self.dt = 0.05
        self.dt_ode = self.dt
        self.M = m # mass of the rod
        self.m = mp  # mass attached to tip
        self.L = l
        self.b = b
        self.I0 = (m* l**2)/3 + mp * l**2  #mass moment of inertia
        self.max_torque = max_torque
        self.max_rot_speed = 8
        self.random_start = random_start
        ALPHA_MAX = 2*np.pi
        obs_high = np.asarray([ALPHA_MAX, np.inf],
                              dtype=np.float64)
        self.observation_space = spaces.Box(-obs_high, obs_high,
                                            dtype=np.float64)

        self.action_space = spaces.Discrete(5)
        self.vis_rate = 1/self.dt
        self.vis = None
        self.states = []
        self._max_episode_steps = int(10/self.dt)
        self.max_episode_steps=self._max_episode_steps
        self.viewer = None


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    

    def reset(self):
        if self.random_start:
            self.state = np.array([np.random.uniform(low = np.pi-0.1, high=np.pi+0.1), np.random.uniform(low = -np.pi, high=np.pi)], dtype=np.float32)
        else:
            self.state = np.array([np.pi, 0], dtype=np.float32)
        # self.state = np.concatenate(([0],np.random.uniform(low = np.pi-0.1, high=np.pi+0.1, size=1),[0],np.random.uniform(low = -np.pi, high=np.pi, size=1)))
        #self.state = np.array([np.random.uniform(low = np.pi-0.1, high=np.pi+0.1), np.random.uniform(low = -np.pi, high=np.pi)], dtype=np.float32)
        self.states.append(self.state)
        self.episode_steps = 0
        self.last_u = None

        return self.state

    def reset_random(self):
        # self.state = np.concatenate(([0],np.random.uniform(low = np.pi-0.1, high=np.pi+0.1, size=1),[0],np.random.uniform(low = -np.pi, high=np.pi, size=1)))
        self.state = np.array([np.random.uniform(low =-2*np.pi, high=2*np.pi), np.random.uniform(low = -5*np.pi, high=5*np.pi)], dtype=np.float32)
        self.states.append(self.state)
        self.vis = None
        self.episode_steps = 0
        return self.state

    def reset_up(self):
        self.state = np.array([np.random.uniform(low =-0.1, high=0.1), np.random.uniform(low = -np.pi, high=np.pi)], dtype=np.float32)
        self.states.append(self.state)
        self.vis = None
        self.episode_steps = 0
        return self.state
    
    def calc_reward(self,u):
        alpha, alpha_d = self.state
        alpha2 =self._angle_normalize(alpha)
        # rew = np.cos(alpha) - 0.01*np.abs(alpha_d)
        r = -(alpha2**2 + 0.1*alpha_d**2 + 0.001 * u**2) # From GYM 
        return r

    def step(self, a):
        g = self.g
        L = self.L
        M = self.M
        m = self.m
        I0 = self.I0
        b = self.b
        if a == 0:
            u = [-self.max_torque]
        elif a == 1:
            u = [-self.max_torque/2]            
        elif a == 2:
            u = [0.0]
        elif a == 3:
           u = [self.max_torque/2]
        elif a == 4:
           u = [self.max_torque]
        
        self.last_u = u[0]  # for rendering
        alpha, alpha_d = self.state
        alpha = self._angle_normalize(alpha)
        alpha_dd = ((m+M/2)*g*L*np.sin(alpha) - b*alpha_d + u[0])/I0
        alpha_d_ = alpha_d + alpha_dd * self.dt
        alpha_ = alpha + alpha_d_*self.dt
        alpha_d_ = np.clip(alpha_d_, -self.max_rot_speed, self.max_rot_speed)
        reward = self.calc_reward(u[0])
        self.state = np.array([alpha_, alpha_d_])
        self.states.append(self.state)
        #reward = 1 - abs(alpha)/np.pi- abs(alpha_d_)/(20*np.pi)
        self.episode_steps += 1
        done = self.episode_steps > self._max_episode_steps
        obs_noise = np.array([np.random.randn()/100, np.random.randn()/100], dtype=np.float32)
        obs_noise = 0
        done=False
        return self.state+obs_noise, reward, done, {}

    def render(self, mode="human"):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, 0.2)
            rod.set_color(0.8, 0.3, 0.3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(0.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1.0, 1.0)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u is not None:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def _angle_normalize(self, x):
        norm_ang = ((x+np.pi) % (2*np.pi)) - np.pi
        return norm_ang

