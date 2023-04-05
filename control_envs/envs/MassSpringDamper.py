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
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
import time
from os import path


class MassEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self,m=1,k=10,c=0.1):
        super(gym.Env).__init__()
        self._m = m  # mass in kg
        self._k = k # spring constant N/m
        self._c = c
        #, m=1,k=1,c=1
        self.dt = 0.02
        self.A = np.array([[1, self.dt], [-self.dt*self._k/self._m , (1-self.dt*self._c/self._m)]], dtype=np.float32)
        self.B = np.array((0, self.dt/self._m))
        self.max_force = 50.0 # [N}]
        self.max_speed = 10.0# [m/s]
        self.max_pos = 5.0 # [m]
        self.min_pos= -5.0
        self.x_threshold = 3
        self.length = 0.5
        self.width = 0.5

        obs_high = np.asarray([self.max_pos, self.max_speed],
                              dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high,
                                            dtype=np.float32)
        self.action_space = spaces.Box(low=-self.max_force,
                                       high=self.max_force,
                                       shape=(1,),
                                       dtype=np.float32)
        self.seed()

        self.states = []
        self._max_episode_steps = int(5/self.dt)
        self.max_episode_steps=self._max_episode_steps
        self.viewer = None
        self.last_f = 0


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    
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
        # self.state = np.array([0,0])
        self.states.append(self.state)
        # self.vis = None
        self.episode_steps = 0
        # print('Reset ...')
        return self.state
    
    
    def calc_reward(self, f): # try to get to the x=0.2 position with zero velocity
        x, x_d = self.state
        r = -((x-1)**2 + 0.1*x_d**2 + 0.01*(f**2)) #
        return r

    def step(self, f):
        f[0] = np.clip(f[0], -self.max_force, self.max_force)
        x_, x_d_ = self.A@self.state + self.B*f[0]
        x_d_ = np.clip(x_d_, -self.max_speed, self.max_speed)
        x_= np.clip(x_, self.min_pos, self.max_pos)
        self.last_f = f[0]
        reward = self.calc_reward(f[0])
        self.state = np.array([x_, x_d_])
        self.states.append(self.state)
        self.episode_steps += 1
        done = False #self.episode_steps > self._max_episode_steps
        obs_noise = np.array([np.random.randn()/100, np.random.randn()/100], dtype=np.float32)
        obs_noise = 0
        done=False
        return self.state+obs_noise, reward, done, {}

    def render(self, mode="human"):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        cartwidth = 30.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            #import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth, cartwidth, cartheight, -cartheight
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            cart.set_color(0.1, 0.1, 0.7)
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            
            self.track = rendering.Line((0,carty-cartheight), (screen_width,carty-cartheight))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)
            
 
            fname_spring = path.join(path.dirname(__file__), "assets/spring.png")
            fname_force = path.join(path.dirname(__file__), "assets/arrow.png")
            self.force_img = rendering.Image(fname_force, 1, 1)
            self.force_img.set_color(0.1, 0.1, 0.7)

            self.force_imgtrans = rendering.Transform()
            self.force_img.add_attr(self.force_imgtrans)
#            self.viewer.add_geom(self.force_img)

  
            
            self.spring_img = rendering.Image(fname_spring, 1, 1)
            self.spr_imgtrans = rendering.Transform()
            self.spring_img.add_attr(self.spr_imgtrans)
#            self.viewer.add_geom(self.spring_img)

            
        self.viewer.add_onetime(self.spring_img)
        self.viewer.add_onetime(self.force_img)
        if self.state is None:
            return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.spr_imgtrans.set_translation(cartx/2,carty) #pixels
        self.force_imgtrans.set_translation(cartx,carty+50) #pixels
        self.force_imgtrans.scale = (self.last_f*3,30) #pixels
        self.spr_imgtrans.scale = (cartx,50)
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

#env  = MassEnv()
#obs = env.reset()
#env.render()

#time.sleep(1)
#for i in range(100):
#    env.step(env.action_space.sample())
#    # time.sleep(0.1)
#    env.render()
#env.close()

