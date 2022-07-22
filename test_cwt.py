# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 11:13:41 2022

@author: cordillet
"""
import pywt
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from pyIMU.importData import from_c3d, resample, getEvents

from scipy import signal

fs=200
s=resample(trial.DataRaw['left_foot'].sel(channel='ACC_X'),fs)

time=s.time.to_numpy()

B=trial.VideoEvent.query("label=='FOG_begin'").time.to_numpy()
E=trial.VideoEvent.query("label=='FOG_end'").time.to_numpy()

            
plt.figure()
plt.plot(time, s.to_numpy())
for b,e in zip(B,E):
    plt.axvspan(b,e,facecolor='g', alpha=0.3)
plt.xticks(time[0::fs*10])
plt.show()


#CWT using pywt
scales=np.arange(0.5,200.5)
dt=1/fs
print(scales)
coefs,freqs=pywt.cwt(s.to_numpy(), scales, 'cgau4', dt)
print(freqs[-2])
plt.figure()
plt.imshow(abs(coefs), interpolation='bilinear',
           aspect='auto', vmax=abs(coefs).max(), vmin=-abs(coefs).max())
# for b,e in zip(B,E):
#     plt.axvspan(b*100,e*100,facecolor='g', alpha=0.3)
plt.gca().invert_yaxis()
plt.xticks(np.arange(0,len(time),100*10),time[0::100*10])
plt.show()

  
plt.yticks(np.arange(0, coefs.shape[0]),np.arange(0.5,1,0.1))
plt.xticks(np.arange(0,len(time[0:8000]),1000),time[0:8000:1000])










