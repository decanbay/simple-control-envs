'''

Pendulum Environment

Author: Deniz Canbay

'''
import importlib
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding


class PendulumEnv(gym.Env):
    def __init__(self):
        super(gym.Env).__init__()
        self.g = 9.807
        self.dt = 0.05
        self.dt_ode = self.dt
        self.M = 1.0 # mass of the rod
        self.m = 0*0.05*1  # mass attached to tip
        self.L = 1.0
        self.b = 0*0.005
        self.I0 = (self.M*self.L**2)/3 + self.m*self.L**2  #mass moment of inertia
        self.max_torque = 2.0
        self.max_rot_speed = 8
        ALPHA_MAX = 2*np.pi
        obs_high = np.asarray([ALPHA_MAX, np.inf],
                              dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high,
                                            dtype=np.float32)
        self.action_space = spaces.Box(low=-self.max_torque,
                                       high=self.max_torque,
                                       shape=(1,),
                                       dtype=np.float32)
        self.vis_rate = 1/self.dt
        self.vis = None
        self.L_vis = self.L
        self.states = []
        self._max_episode_steps = int(10/self.dt)
        self.max_episode_steps=self._max_episode_steps


    def reset(self):
        # self.state = np.concatenate(([0],np.random.uniform(low = np.pi-0.1, high=np.pi+0.1, size=1),[0],np.random.uniform(low = -np.pi, high=np.pi, size=1)))
        self.state = np.array([np.random.uniform(low = np.pi-0.1, high=np.pi+0.1), np.random.uniform(low = -np.pi, high=np.pi)], dtype=np.float32)
        self.states.append(self.state)
        self.vis = None
        self.episode_steps = 0
        # print('Reset ...')
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
        g = self.g
        L = self.L
        M = self.M
        m = self.m
        I0 = self.I0
        b = self.b
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

    def render(self, mode='human'):
        theta=0
        if self.vis is None:
            self.__set_vis()
        else:
            alpha, _ = self.state
            vector = self.vis.vector
            pend_ax = vector(self.L_vis*np.sin(-alpha)*np.cos(theta),
                             self.L_vis*np.cos(-alpha),
                             -self.L_vis*np.sin(-alpha)*np.sin(theta))
            self.pend.pos = self.vis.arm_start_pos
            self.pend.axis = pend_ax
            self.vis.rate(self.vis_rate)

    def __set_vis(self):
        theta=0
        self.vis = importlib.import_module('vpython')
        self.vis.scene.width = 800
        self.vis.scene.height = 600
        self.vis.scene.background = self.vis.color.gray(0.95)
        self.vis.scene.title = 'simple pendulum'
        self.vis.scene.range = 0.6
        vector = self.vis.vector
        camera = self.vis.scene.camera
        camera.pos = vector(0, 0.2, 1.2)
        camera.axis = vector(0, -0.3, -0.9)
        alpha, _ = self.state
        pend_ax = vector(self.L_vis*np.sin(-alpha)*np.cos(theta),
                         self.L_vis*np.cos(-alpha), -self.L_vis*np.sin(-alpha)*np.sin(theta))
        self.vis.arm_start_pos = vector(0, 0.107, -0.002)
        self.pend = self.vis.cylinder(pos=self.vis.arm_start_pos, axis=pend_ax,
                                      radius=0.01, color=self.vis.color.blue)
        self.vis.rate(self.vis_rate)

    def _angle_normalize(self, x):
        norm_ang = ((x+np.pi) % (2*np.pi)) - np.pi
        return norm_ang


class PendulumDiscEnv(gym.Env):
    def __init__(self):
        super(gym.Env).__init__()
        self.g = 9.807
        self.dt = 0.05
        self.dt_ode = self.dt
        self.M = 1.0 # mass of the rod
        self.m = 0*0.05*1  # mass attached to tip
        self.L = 1.0
        self.b = 0*0.005
        self.I0 = (self.M*self.L**2)/3 + self.m*self.L**2  #mass moment of inertia
        self.max_torque = 2.0
        self.max_rot_speed = 8
        ALPHA_MAX = 2*np.pi
        obs_high = np.asarray([ALPHA_MAX, np.inf],
                              dtype=np.float64)
        self.observation_space = spaces.Box(-obs_high, obs_high,
                                            dtype=np.float64)

        self.action_space = spaces.Discrete(5)
        self.vis_rate = 1/self.dt
        self.vis = None
        self.L_vis = self.L
        self.states = []
        self._max_episode_steps = int(10/self.dt)
        self.max_episode_steps=self._max_episode_steps


    def reset(self):
        # self.state = np.concatenate(([0],np.random.uniform(low = np.pi-0.1, high=np.pi+0.1, size=1),[0],np.random.uniform(low = -np.pi, high=np.pi, size=1)))
        self.state = np.array([np.random.uniform(low = np.pi-0.1, high=np.pi+0.1), np.random.uniform(low = -np.pi, high=np.pi)], dtype=np.float32)
        self.states.append(self.state)
        self.vis = None
        self.episode_steps = 0
        # print('Reset ...')
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

    def render(self, mode='human'):
        theta=0
        if self.vis is None:
            self.__set_vis()
        else:
            alpha, _ = self.state
            vector = self.vis.vector
            pend_ax = vector(self.L_vis*np.sin(-alpha)*np.cos(theta),
                             self.L_vis*np.cos(-alpha),
                             -self.L_vis*np.sin(-alpha)*np.sin(theta))
            self.pend.pos = self.vis.arm_start_pos
            self.pend.axis = pend_ax
            self.vis.rate(self.vis_rate)

    def __set_vis(self):
        theta=0
        self.vis = importlib.import_module('vpython')
        self.vis.scene.width = 800
        self.vis.scene.height = 600
        self.vis.scene.background = self.vis.color.gray(0.95)
        self.vis.scene.title = 'simple pendulum'
        self.vis.scene.range = 0.6
        vector = self.vis.vector
        camera = self.vis.scene.camera
        camera.pos = vector(0, 0.2, 1.2)
        camera.axis = vector(0, -0.3, -0.9)
        alpha, _ = self.state
        pend_ax = vector(self.L_vis*np.sin(-alpha)*np.cos(theta),
                         self.L_vis*np.cos(-alpha), -self.L_vis*np.sin(-alpha)*np.sin(theta))
        self.vis.arm_start_pos = vector(0, 0.107, -0.002)
        self.pend = self.vis.cylinder(pos=self.vis.arm_start_pos, axis=pend_ax,
                                      radius=0.01, color=self.vis.color.blue)
        self.vis.rate(self.vis_rate)

    def _angle_normalize(self, x):
        norm_ang = ((x+np.pi) % (2*np.pi)) - np.pi
        return norm_ang

