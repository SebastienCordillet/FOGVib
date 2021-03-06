# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 10:57:53 2022

@author: cordillet_s
"""

# from fileinput import filename
from scipy import signal
import numpy as np
from pyomeca import Analogs


def getFOGInstances(filename, R_acc_ap, L_acc_ap, R_gyr_ml, L_gyr_ml, sampleRate):
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
    timeSR=np.arange(0, duration,1/sampleRate)
    time200=np.arange(0, duration,1/200)

    #Resample from 128 to 200Hz
    R_ml=np.interp(time200,timeSR,R_gyr_ml)
    L_ml=np.interp(time200,timeSR,L_gyr_ml)

    sensor_data=np.array([R_ml,L_ml])

    #Low pass filter
    b, a = signal.butter(4, 5/(200/2), 'lowpass')
    sensor_data_f=signal.filtfilt(b, a, sensor_data)

    ##============== Correlation Based Method ==================
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
    
    bool_gyro= abs(y_r) <0.5

    ##================= FFT based method ======================
    #Resample from 128 to 200Hz
    R_ap=np.interp(time200,timeSR,R_acc_ap)
    L_ap=np.interp(time200,timeSR,L_acc_ap)
    #Synchronizing both R and L together after correction for the delay
    t_R_ap=R_ap[round(abs(lag)):-1]
    t_L_ap=L_ap[0:-round(abs(lag))-1]
    
    ratio_L=np.array([],dtype=float)
    ratio_R=np.array([],dtype=float)
    for w in x_r:
        #left
        fft=abs(np.fft.fft(signal.detrend(t_L_ap[w:w+200])))
        freq=np.fft.fftfreq(fft.shape[-1])*200
        walk_mask=np.logical_and(abs(freq)>0, abs(freq)<3)
        fog_mask=np.logical_and(abs(freq)>3, abs(freq)<10)
        ratio_L=np.append(ratio_L,sum(fft.real[fog_mask])**2/sum(fft.real[walk_mask])**2)
        #right
        fft=abs(np.fft.fft(signal.detrend(t_R_ap[w:w+200])))
        ratio_R=np.append(ratio_R,sum(fft.real[fog_mask])**2/sum(fft.real[walk_mask])**2)
    
    bool_acc=np.logical_or(ratio_L>10,ratio_R>10)

    bool_FOG=np.logical_and(bool_acc,bool_gyro)
    
    ##====== Classifier FOG episodes ==================
    # at the beggining, we have not fog episode 
    n_veryshort_fog=0
    n_short_fog=0
    n_long_fog=0
    n_very_long_fog=0
    
    # at first second
    if bool_FOG[0]:
        n_count=1 # to count second of freezing
    else:
        n_count=0
    
    for b in bool_FOG[1:]:
        if b: #begining of episode
            n_count+=1
        elif not b and n_count>0: #???ending of episode
            if n_count ==1:
                n_veryshort_fog+=1
            elif n_count >1 and n_count<=5:
                n_short_fog+=1
            elif n_count>5 and n_count<=30:
                n_long_fog+=1
            elif n_count>30:
                n_very_long_fog+=1
            n_count=0
    
    perc_of_FOG=round(100*sum(bool_FOG)/len(bool_FOG))
    
    # Add variable to dict of results
    IFOG["n_veryshort_fog"]=n_veryshort_fog
    IFOG["n_short_fog"]=n_short_fog
    IFOG["n_long_fog"]=n_long_fog
    IFOG["n_very_long_fog"]=n_very_long_fog
    IFOG["perc_of_FOG"]=perc_of_FOG
    
    return(IFOG)

def FOGepisodes(fileC3D):
    analogs=Analogs.from_c3d(fileC3D)
    
    
                

    







