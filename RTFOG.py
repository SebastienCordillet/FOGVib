# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 14:47:01 2022

@author: cordillet
"""
import pandas as pd
import numpy as np
from scipy import signal

class RTFOG:
    """
    Detecter de FOG en temp réel
    """
    
    def __init__(self, windowsLength=200, overlay=20):
        self.windowsLength=windowsLength #ms
        self.overlay=overlay #ms
        self.windowCount=0
        self.data=pd.DataFrame(data={
            'left_thigh_wz':[],
            'left_thigh_ax':[],
            'left_thigh_ay':[],
            'right_thigh_wz':[],
            'right_thigh_ax':[],
            'right_thigh_ay':[],
            'left_shank_wz':[],
            'left_shank_ax':[],
            'left_shank_ay':[],
            'right_shank_wz':[],
            'right_shank_ax':[],
            'right_shank_ay':[]
            })
        self.flags=pd.DataFrame(data={
            'windowId':[],
            'cor_thigh':[],#maximal correlation
            'cor_shank':[],#maximal correlation
            'left_thigh_ax':[],#freezing ratio
            'left_thigh_ay':[],#freezing ratio
            'right_thigh_ax':[],#freezing ratio
            'right_thigh_ay':[],#freezing ratio
            'left_shank_ax':[],#freezing ratio
            'left_shank_ay':[],#freezing ratio
            'right_shank_ax':[],#freezing ratio
            'right_shank_ay':[]#freezing ratio
            })
        def maxCorrelation(self, threshold=0.5):
            if len(self.data)!=self.windowsLength:
                return
            ## Thigh 
            cor=signal.correlate(self.data['left_thigh_wz'].values,
                                 self.data['right_thigh_wz'].values, mode="full")
            lags = signal.correlation_lags(self.data['left_thigh_wz'].size,
                                           self.data['right_thigh_wz'].size, mode="full")
            lag = lags[np.argmax(cor)]

            #Synchronizing both R and L together after correction for the delay
            t_R_ml=self.data['left_thigh_wz'].values[round(abs(lag)):-1]
            t_L_ml=self.data['right_thigh_wz'].values[0:-round(abs(lag))-1]

            t_cor=np.corrcoef(t_R_ml,t_L_ml)[0,1]

            ## Shank 
            cor=signal.correlate(self.data['left_shank_wz'].values,
                                 self.data['right_shank_wz'].values, mode="full")
            lags = signal.correlation_lags(self.data['left_shank_wz'].size,
                                           self.data['right_shank_wz'].size, mode="full")
            lag = lags[np.argmax(cor)]

            #Synchronizing both R and L together after correction for the delay
            t_R_ml=self.data['left_shank_wz'].values[round(abs(lag)):-1]
            t_L_ml=self.data['right_shank_wz'].values[0:-round(abs(lag))-1]

            s_cor=np.corrcoef(t_R_ml,t_L_ml)[0,1]            
            
            self.flags.iloc[-1,].loc['cor_thigh']=(t_cor<threshold)
            self.flags.iloc[-1,].loc['cor_shank']=(t_cor<threshold)
            
            
        def FreezingRatio(self, inputStr, threshold=10):
            fft=abs(np.fft.fft(signal.detrend(self.data[inputStr].value)))
            freq=np.fft.fftfreq(fft.shape[-1])*self.windowsLength
            walk_mask=np.logical_and(abs(freq)>0, abs(freq)<3)
            fog_mask=np.logical_and(abs(freq)>3, abs(freq)<10)
            FR=sum(fft.real[fog_mask])**2/sum(fft.real[walk_mask])**2
            
            self.flags.iloc[-1,].loc[inputStr]=(FR>threshold)
        
        def allFreezingRatio(self):
            listing=[
                'left_thigh_ax',
                'left_thigh_ay',
                'right_thigh_ax',
                'right_thigh_ay',
                'left_shank_ax',
                'left_shank_ay',
                'right_shank_ax',
                'right_shank_ay'
                ]
            
            for l in listing:
                FreezingRatio(l)
            
            
                