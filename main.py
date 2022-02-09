# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 14:34:53 2022

@author: cordillet_s
"""

from fileinput import filename
from scipy import signal
from pyomeca import Analogs, Rototrans, Angles
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import plotly
import plotly.graph_objs as go
from plotly.offline import plot
import plotly.io as pi

#pour faire de la viz dans spyder
pio.renderers.default='browser'

def getFOGInstances(filname, R_acc_ap, L_acc_ap, R_gyr_ml, L_gyr_ml, sampleRate):
    """ Detect FOGs using FFT-based method with accelerometer data
    Detect FOGs using correlation-based methods with gyroscope data
    If both methods detect a FOG, register it as  FOG
    Break into very short, short, and long FOG episodes """

    #Initailize dict with results 
    IFOG={
        'filename': filename
    }

    # Initial time Series
    duration=len(R_gyr_ml)/sampleRate
    timeSR=np.arrange(0, duration,1/sampleRate)
    time200=np.arrange(0, duration,1/200)

    #Resample from 128 to 200Hz
    R_ml=np.interp(time200,timeSR,R_gyr_ml)
    L_ml=np.interp(time200,timeSR,L_gyr_ml)

    sensor_data=np.array([R_ml,L_ml])

    #Low pass filter
    b, a = signal.butter(4, 5/(200/2), 'lowpass')
    sensor_data_f=signal.filtfilt(b, a, sensor_data)

    ## Correlation Based Method
    #R and L angular velocities cross-correlation to find a delay between two
    cor=signal.correlate(sensor_data_f.T[0],sensor_data_f.T[1], mode="full")
    lags = signal.correlation_lags(sensor_data_f.T[0].size, sensor_data_f.T[1].size, mode="full")
    lag = lags[np.argmax(cor)]

    #Synchronizing both R and L together after correction for the delay
    t_R_ml=sensor_data_f.T[round(abs(lag)):-1,0]
    t_L_ml=sensor_data_f.T[0:-round(abs(lag))-1,1]

    #Test on trial duration
    if len(t_R_ml)<200*5:
        raise ValueError('Error: There is not enough data to calculate FOG. Please ensure each bout is longer than 10 seconds or sample more often')

    y_r=np.array([],dtype=float)
    x_r=np.arange(0,len(t_R_ml)-400,200)
    for w in x_r:
        y_r=np.append(y_r,np.corrcoef(t_R_ml[w:w+200],t_L_ml[w:w+200])[0,1])
        




