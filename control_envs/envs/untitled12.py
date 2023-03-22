#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 14:50:27 2023

@author: deniz
"""
import numpy as np

class Pendulum():
    def __init__(self,m=1,k=10,c=0.1, dt=0.01):
        self.__m = m  # mass in kg
        self.__k = k # spring constant N/m
        self.__c = c
        self.dt = dt
        self.A = np.array([[1, self.dt], [-self.dt*self.k/self.m , (1-self.dt*self.c/self.m)]], dtype=np.float32)
        self.B = np.array((0, self.dt/self.m))
    
    @property
    def m(self):
        return self.__m
    
    @m.setter
    def m(self, mval):
        self.__m = mval
        self.A , self.B = self.recalc()

    
    @property
    def c(self):
        return self.__c
        
    @c.setter
    def c(self, cval):
        print('c Setter')
        self.__c = cval
        self.A , self.B = self.recalc()
  

    @property
    def k(self):
        return self.__k
   
    @k.setter
    def k(self, kval):
        self.__k = kval
        self.A , self.B = self.recalc()

    def recalc(self):
        A = np.array([[1, self.dt], [-self.dt*self.k/self.m , (1-self.dt*self.c/self.m)]], dtype=np.float32)
        B = np.array((0, self.dt/self.m))
        return A, B
        
     



p = Pendulum()

print(p.m)
print(p.B)
print(p.A)
p.c = 0
print(p.A)
print(p.B)