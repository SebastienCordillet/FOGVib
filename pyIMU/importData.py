# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 14:11:20 2022

@author: cordillet_s
"""
from pyomeca import Analogs
import numpy as np
import pandas as pd
from ezc3d import c3d


def from_c3d(filename, chanelName, acc=True, gyr=True, mag=True, name=""):
    #list des channels à importer
    channels=[]
    if acc:
        channels.append(chanelName+'_ACC_X')
        channels.append(chanelName+'_ACC_Y')
        channels.append(chanelName+'_ACC_Z')
    if gyr:
        channels.append(chanelName+'_GYRO_X')
        channels.append(chanelName+'_GYRO_Y')
        channels.append(chanelName+'_GYRO_Z')
    if mag:
        channels.append(chanelName+'_MAG_X')
        channels.append(chanelName+'_MAG_Y')
        channels.append(chanelName+'_MAG_Z')       
   
    #import du c3d à partir du nom du fichier
    analogs=Analogs.from_c3d(filename, usecols=channels)
    
    #set channels names    
    # if name is not None:
    newChannels = [w.replace(chanelName+"_", name) for w in channels]
    analogs.coords["channel"]=newChannels
        
    return(analogs)

def resample(analogs, rate=120, times=None):
    #vecteur temps attendu
    if times is None:
        times=[0, analogs['time'].values[-1]]
        
    time = np.arange(start=times[0], stop=times[1], step=1 / rate)
    
    #utilisation de time_noramlize from pyomeca
    analogs=analogs.meca.time_normalize(time_vector=time)
    analogs=analogs.assign_attrs({'rate':rate})
    return(analogs)

def acc_fromAnalogs(analogs, name="", listChannel=['ACC_X','ACC_Y','ACC_Z']):
    channels=[name+channel for channel in listChannel]
    return(analogs.sel(channel=channels).values.T)

def gyr_fromAnalogs(analogs, name="", listChannel=['GYRO_X','GYRO_Y','GYRO_Z']):
    channels=[name+channel for channel in listChannel]
    return(analogs.sel(channel=channels).values.T)

def getEvents(filec3d, part='EVENT'):
    c=c3d(filec3d)
    labels=c['parameters'][part]['LABELS']['value']
    time=c['parameters'][part]['TIMES']['value'][1,:]

    
    try:
        fr_marker=c['parameters']['POINT']['RATE']['value'].astype(int)
    except Exception:
        pass
    
    try:
        fr_analog=c['parameters']['ANALOG']['RATE']['value'].astype(int)
    except Exception:
        pass
    
    try:
        context=c['parameters'][part]['CONTEXTS']['value']
    except Exception:
        pass
    
    events=pd.DataFrame(data={
        'label':labels,
        'time': time
        })
    

    if 'fr_marker' in locals():
        events['frame_marker']= np.array(time*fr_marker, dtype=int)
        
    if 'fr_analog' in locals():
        events['frame_analog']= np.array(time*fr_analog, dtype=int)
        
    if 'context' in locals():
        events.assign(1, 'Context', context)
        
    return(events)
    