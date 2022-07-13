# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 14:47:01 2022

@author: cordillet
"""
import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
pio.renderers.default='browser'

from pyIMU.importData import from_c3d, resample, getEvents

class RTFOG:
    """
    Detecter de FOG en temp réel
    """
    def __init__(self, windowsLength=200, overlay=200):
        self.windowsLength=windowsLength #ms
        self.overlay=overlay #ms si windowsLength=overlay pas de chevauchement
        self.windowCount=0
        self.currentFrame=0
        self.nFrames=0
        self.timeArray=[]
        self.timeWindows=[]
        self.data=pd.DataFrame(data={
            'left_foot_wz':[],
            'left_foot_ax':[],
            'left_foot_ay':[],
            'right_foot_wz':[],
            'right_foot_ax':[],
            'right_foot_ay':[],
            'left_shank_wz':[],
            'left_shank_ax':[],
            'left_shank_ay':[],
            'right_shank_wz':[],
            'right_shank_ax':[],
            'right_shank_ay':[]
            })
        self.flags=pd.DataFrame(data={
            'windowId':[],
            'videoFOG':[],
            'cor_foot':[],#maximal correlation
            'cor_shank':[],#maximal correlation
            'left_foot_ax':[],#freezing ratio
            'left_foot_ay':[],#freezing ratio
            'right_foot_ax':[],#freezing ratio
            'right_foot_ay':[],#freezing ratio
            'left_shank_ax':[],#freezing ratio
            'left_shank_ay':[],#freezing ratio
            'right_shank_ax':[],#freezing ratio
            'right_shank_ay':[]#freezing ratio
            })
        
        self.DataRaw={} #will contains imu raw data
        self.ResampleRate=200
        self.DataResampled={} #will contains imu resampled data
        self.filterOrder=4
        self.filtercutOff=5
        self.DataFiltered={}#will contains imu filtered data
        
    def maxCorrelation(self, threshold=0.5, isStatic=False):
        if len(self.data)!=self.windowsLength:
            return
        ## foot 
        cor=signal.correlate(self.data['left_foot_wz'].values,
                             self.data['right_foot_wz'].values, mode="full")
        lags = signal.correlation_lags(self.data['left_foot_wz'].size,
                                       self.data['right_foot_wz'].size, mode="full")
        lag = lags[np.argmax(cor)]

        #Synchronizing both R and L together after correction for the delay
        t_R_ml=self.data['left_foot_wz'].values[round(abs(lag)):-1]
        t_L_ml=self.data['right_foot_wz'].values[0:-round(abs(lag))-1]

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
        
        if isStatic:
            self.flags.iloc[-1, self.flags.columns.get_loc('cor_foot')]=np.nan
            self.flags.iloc[-1, self.flags.columns.get_loc('cor_shank')]=np.nan
        else:
            self.flags.iloc[-1, self.flags.columns.get_loc('cor_foot')]=abs(t_cor) #previsouly(t_cor<threshold)
            self.flags.iloc[-1, self.flags.columns.get_loc('cor_shank')]=abs(s_cor) #previsouly(s_cor<threshold)
        
    def FreezingRatio(self, inputStr, threshold=10, isStatic=False):
        fft=abs(np.fft.fft(signal.detrend(self.data[inputStr])))
        freq=np.fft.fftfreq(fft.shape[-1])*self.windowsLength
        walk_mask=np.logical_and(abs(freq)>0, abs(freq)<3)
        fog_mask=np.logical_and(abs(freq)>3, abs(freq)<10)
        FR=sum(fft.real[fog_mask])**2/sum(fft.real[walk_mask])**2
        
        if isStatic:
            self.flags.iloc[-1, self.flags.columns.get_loc(inputStr)]=np.nan
        else:
            self.flags.iloc[-1, self.flags.columns.get_loc(inputStr)]=int(FR) # previsouly(FR>threshold)
    
    def allFreezingRatio(self, isStatic=False):
        listing=[
            'left_foot_ax',
            'left_foot_ay',
            'right_foot_ax',
            'right_foot_ay',
            'left_shank_ax',
            'left_shank_ay',
            'right_shank_ax',
            'right_shank_ay'
            ]
        for l in listing: 
            self.FreezingRatio(l, isStatic=isStatic)
            
    def isVideoFOG(self):
        startFOG=self.VideoEvent.query("label=='FOG_begin'").time.to_numpy()
        endFOG=self.VideoEvent.query("label=='FOG_end'").time.to_numpy()
        
        currentTimes=self.timeArray[self.currentFrame:self.currentFrame+self.windowsLength]
        self.flags.iloc[-1,self.flags.columns.get_loc('videoFOG')]= np.sum([(currentTimes > B)*(currentTimes<E) for B,E in zip(startFOG,endFOG)])/self.windowsLength
        
    def isStatic(self, threshold=0.04, cutoff=6):
        acc_left_shank_Y_sd=self.data.left_shank_ay.std()
        acc_left_shank_X_sd=self.data.left_shank_ax.std()
        
        acc_left_foot_Y_sd=self.data.left_foot_ay.std()
        acc_left_foot_X_sd=self.data.left_foot_ax.std()
        
        acc_right_shank_Y_sd=self.data.right_shank_ay.std()
        acc_right_shank_X_sd=self.data.right_shank_ax.std()
        
        acc_right_foot_Y_sd=self.data.right_foot_ay.std()
        acc_right_foot_X_sd=self.data.right_foot_ax.std()
        
        acc_sd=np.array([
            acc_left_shank_Y_sd,
            acc_left_shank_X_sd,
            acc_left_foot_Y_sd,
            acc_left_foot_X_sd,
            acc_right_shank_Y_sd,
            acc_right_shank_X_sd,
            acc_right_foot_Y_sd,
            acc_right_foot_X_sd
            ])
        acc_isStatic=np.sum(acc_sd<threshold)
        return(acc_isStatic>cutoff)
        
        
            
    def plotFlags(self):
        data2plot=self.flags.loc[:, self.flags.columns != 'windowId'].to_numpy(dtype=float).transpose()
        fig=px.imshow(data2plot,labels=dict(x="Time",y=""),color_continuous_scale=["lightgreen", "firebrick"], 
                      y=['videoFOG',
                         'cor_foot',
                         'cor_shank',
                         'left_foot_ax',
                         'left_foot_ay',
                         'right_foot_ax',
                         'right_foot_ay',
                         'left_shank_ax',
                         'left_shank_ay',
                         'right_shank_ax',
                         'right_shank_ay'])
        fig.update_xaxes(side="top", 
                         tickmode='array',
                         tickvals=np.arange(0, self.windowCount)[0::10],
                         ticktext=self.timeWindows.astype('str')[0::10])
        fig.layout.coloraxis.showscale = False
        fig.show()
        
    def timeFlags(self):
        
        fig=px.line(self.flags.drop(['windowId', 'videoFOG'],axis=1))
        B=self.VideoEvent.query("label=='FOG_begin'").time.to_numpy()
        E=self.VideoEvent.query("label=='FOG_end'").time.to_numpy()
        for b,e in zip(B,E):
            fig.add_vrect(
                x0=b,
                x1=e,
                fillcolor="green",opacity=0.25, line_width=0
                )
        fig.show()
        
    def plotCurrentWindow(self,var):
        fig=px.line(self.data[var])
        fig.show()


        
    def importC3D(self, filename, chanelNames, newCNames=None):
        if newCNames is None:
            newCNames=['left_shank','left_foot','right_shank','right_foot']
        
        for chanelName, newCName in zip(chanelNames,newCNames):
            self.DataRaw[newCName]=from_c3d(filename, chanelName, acc=True, gyr=True, mag=False,name='')
        self.nFrames=len(self.DataRaw[newCName].isel(channel=1).values)
        self.VideoEvent=getEvents(filename)
        
    def resampleData(self):
        for key,imu in self.DataRaw.items():
            self.DataResampled[key]=resample(imu, self.ResampleRate)
        self.timeArray=self.DataResampled[key].isel(channel=1).time.values
            
    def filterData(self):
        for key,imu in self.DataResampled.items():
            self.DataFiltered[key]=imu.meca.low_pass(order=self.filterOrder, cutoff=self.filtercutOff, freq=self.ResampleRate)
            
            
    def setCurrentFlags(self):
        self.maxCorrelation(isStatic=self.isStatic())
        self.allFreezingRatio(isStatic=self.isStatic())
        self.isVideoFOG()
        self.flags.iloc[-1, self.flags.columns.get_loc("windowId")]=float(self.timeArray[self.currentFrame])
        
    def updateData(self):
        if self.currentFrame+self.windowsLength>self.nFrames:
            print('Attention /!\  : pas d\'update on arrive à la fin de la série')
            return
            
        w_begin=self.currentFrame
        w_end=self.currentFrame+self.windowsLength
        
        self.timeWindows=np.append(self.timeWindows,self.timeArray[self.currentFrame])
        
        #update data
        self.data['left_foot_wz']=self.DataFiltered['left_foot'].sel(channel='GYRO_Z').values[w_begin:w_end]
        self.data['left_foot_ax']=self.DataFiltered['left_foot'].sel(channel='ACC_X').values[w_begin:w_end]
        self.data['left_foot_ay']=self.DataFiltered['left_foot'].sel(channel='ACC_Y').values[w_begin:w_end]
        self.data['left_shank_wz']=self.DataFiltered['left_shank'].sel(channel='GYRO_Z').values[w_begin:w_end]
        self.data['left_shank_ax']=self.DataFiltered['left_shank'].sel(channel='ACC_X').values[w_begin:w_end]
        self.data['left_shank_ay']=self.DataFiltered['left_shank'].sel(channel='ACC_Y').values[w_begin:w_end]
        
        self.data['right_foot_wz']=-1*self.DataFiltered['right_foot'].sel(channel='GYRO_Z').values[w_begin:w_end]
        self.data['right_foot_ax']=self.DataFiltered['right_foot'].sel(channel='ACC_X').values[w_begin:w_end]
        self.data['right_foot_ay']=self.DataFiltered['right_foot'].sel(channel='ACC_Y').values[w_begin:w_end]
        self.data['right_shank_wz']=-1*self.DataFiltered['right_shank'].sel(channel='GYRO_Z').values[w_begin:w_end]
        self.data['right_shank_ax']=self.DataFiltered['right_shank'].sel(channel='ACC_X').values[w_begin:w_end]
        self.data['right_shank_ay']=self.DataFiltered['right_shank'].sel(channel='ACC_Y').values[w_begin:w_end]
        
        #increment currentFrame to next window
        self.currentFrame += self.overlay
        self.windowCount +=1
        
    def loopProcessing(self):
        ww=np.arange(0,len(self.timeArray)-self.windowsLength,self.overlay)
        
        for _ in ww:
            self.updateData()
            self.flags=self.flags.append(pd.Series(), ignore_index=True)
            self.setCurrentFlags()
            # if self.windowCount==13 :
            #     self.plotCurrentWindow(['left_foot_wz','right_foot_wz'])
                

#test sur des données 
if __name__=="__main__":
    file='./dataset/FOG_sim_v2/sujet1K/simFOG.c3d'
    trial=RTFOG()
    trial.importC3D(file, chanelNames=['Left_Rectus Femoris',
                                       'Left_Vastus Lateralis',
                                       'Left_Tibialis Anterior',
                                       'Right_Rectus Femoris'])
    trial.resampleData()
    trial.filterData()
    trial.loopProcessing()

    # trial.plotFlags()
    trial.timeFlags()