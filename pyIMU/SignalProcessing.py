# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 14:19:40 2022

@author: cordillet_s
"""

import numpy as np
from scipy import signal

def getNorm(v):
    if v.shape[1]==3:
        return(np.sqrt(v[:,0]*v[:,0] + v[:,1]*v[:,1] + v[:,2]*v[:,2]))
    
def lowpass(c, filtCutOff=5,rate=120, order=4):
    samplePeriod=1/rate
    b, a = signal.butter(order, (2*filtCutOff)/(1/samplePeriod), 'lowpass')
    c_filt = signal.filtfilt(b, a, c, padtype = 'odd', padlen=3*(max(len(b),len(a))-1))
    return(c_filt)

def signFilt(s,**kwargs):
    s_filt=np.apply_along_axis(lowpass,0,s,**kwargs)
    return(s_filt)