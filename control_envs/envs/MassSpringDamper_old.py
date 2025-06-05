'''

Mass Spring Damper Environment

Author: Deniz Canbay

'''

import importlib
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding


class MassEnv(gym.Env):
    def __init__(self,m=1,k=10,c=0.1):
        super().__init__()
        self._m = m  # mass in kg
        self._k = k # spring constant N/m
        self._c = c
        #, m=1,k=1,c=1
        self.dt = 0.01
        self.A = np.array([[1, self.dt], [-self.dt*self._k/self._m , (1-self.dt*self._c/self._m)]], dtype=np.float32)
        self.B = np.array((0, self.dt/self._m))
        self.max_force = 50.0 # [N}]
        self.max_speed = 10.0# [m/s]
        self.max_pos = 5.0 # [m]
        self.min_pos= -5.0
        obs_high = np.asarray([self.max_pos, self.max_speed],
                              dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high,
                                            dtype=np.float32)
        self.action_space = spaces.Box(low=-self.max_force,
                                       high=self.max_force,
                                       shape=(1,),
                                       dtype=np.float32)
        self.vis_rate = 1/self.dt
        self.vis = None
        self.states = []
        self._max_episode_steps = int(10/self.dt)
        self.max_episode_steps=self._max_episode_steps

    # @property
    # def m(self):
    #     print('m set')
    #     return duper(super()).m
    
    # @m.setter
    # def m(self, mval):
    #     duper(super()).m = mval
    #     self.A , self.B = self.recalc()
    #     print('m setter run')
    
    # @property
    # def c(self):
    #     return self.__c
        
    # @c.setter
    # def c(self, cval):
    #     print('c Setter')
    #     self.__c = cval
    #     self.A , self.B = self.recalc()     

    # @property
    # def k(self):
    #     return self.__k
   
    # @k.setter
    # def k(self, kval):
    #     self.__k = kval
    #     self.A , self.B = self.recalc()
    
    def recalc(self):
        A = np.array([[1, self.dt], [-self.dt*self._k/self._m , (1-self.dt*self._c/self._m)]], dtype=np.float32)
        B = np.array((0, self.dt/self._m))
        return A, B

    def reset(self):
        self.A , self.B = self.recalc()
        pos = -1
        noise_magnitude=0.02 
        self.state = np.array([np.random.uniform(low = pos-noise_magnitude,
                                                 high= pos+noise_magnitude),
                               np.random.uniform(low = -noise_magnitude,
                                                 high= noise_magnitude)],
                              dtype=np.float32)
        self.states.append(self.state)
        self.vis = None
        self.episode_steps = 0
        # print('Reset ...')
        return self.state
    
    
    def calc_reward(self, f): # try to get to the x=0.2 position with zero velocity
        x, x_d = self.state
        r = -((x-1)**2 + 0.1*x_d**2 + 0.01*(f**2)) #
        return r

    def step(self, f):
        # f[0] = np.clip(f[0], -self.max_force, self.max_force)
        x_, x_d_ = self.A@self.state + self.B*f[0]
        x_d_ = np.clip(x_d_, -self.max_speed, self.max_speed)
        x_= np.clip(x_, self.min_pos, self.max_pos)
        
        reward = self.calc_reward(f[0])
        self.state = np.array([x_, x_d_])
        self.states.append(self.state)
        self.episode_steps += 1
        done = False #self.episode_steps > self._max_episode_steps
        obs_noise = np.array([np.random.randn()/100, np.random.randn()/100], dtype=np.float32)
        obs_noise = 0
        done=False
        return self.state+obs_noise, reward, done, {}

    def render(self, mode='human'):
        if self.vis is None:
            self.__set_vis()
        else:
            x, _ = self.state
            vector = self.vis.vector
            self.mass.pos = vector(x,0,0)
            self.spring.length = 2.5- self.mass.length/2 +x
            self.vis.rate(self.vis_rate)

    def __set_vis(self):
        self.vis = importlib.import_module('vpython')
        self.vis.scene.width = 1600
        self.vis.scene.height = 1200
        self.vis.scene.background = self.vis.color.gray(0.95)
        self.vis.scene.title = 'Mass Spring Damper System'
        self.vis.scene.range = 0.6
        vector = self.vis.vector
        camera = self.vis.scene.camera
        camera.pos = vector(0, 0.0, 5)
        camera.axis = vector(0, -0.3, -0.9)
        x, _ = self.state
        self.spring = self.vis.helix(pos=vector(-2.5,0,0), axis=vector(1,0,0), radius= 0.02)
        self.spring.thickness = 0.01 #self.k/50
        self.spring.coils = 20
        self.mass = self.vis.box(pos=vector(self.state[0],0,0), length=0.1, height=0.1, width=0.1, color=self.vis.color.blue)
        self.spring.length = 2.5- self.mass.length/2 +x
        self.vis.rate(self.vis_rate)


