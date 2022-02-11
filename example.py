from main import getFOGInstances
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import signal

subjectFolder = './dataset/';
yourOutputDirectory = '/results';
sampleRate = 128;

listFilename=os.listdir(subjectFolder)
filename='fogMetrics.csv'
# df=np.genfromtxt(filename, delimiter=',',names=True)
df=pd.read_csv(subjectFolder+filename)

plt.plot(np.arange(0,len(df)/128,1/128),df)
plt.show()


R_acc_ap=df['R_acc']
L_acc_ap=df['L_acc']
R_gyr_ml=df['R_gyr']
L_gyr_ml=df['L_gyr']

ifogs=[]
for f in listFilename:
    df=pd.read_csv(subjectFolder+f)
    R_acc_ap=df['R_acc']
    L_acc_ap=df['L_acc']
    R_gyr_ml=df['R_gyr']
    L_gyr_ml=df['L_gyr']
    ifogs.append(getFOGInstances(f, R_acc_ap, L_acc_ap, R_gyr_ml, L_gyr_ml, sampleRate))
    plt.plot(np.arange(0,len(df)/128,1/128),df)
    plt.show()
    
# ifog=getFOGInstances(filename, R_acc_ap, L_acc_ap, R_gyr_ml, L_gyr_ml, sampleRate)
# print(ifog)
print(ifogs)



## ============ EFFET de la sync ==============
filename='fogMetrics.csv'
df=pd.read_csv(subjectFolder+filename)

R_acc_ap=df['R_acc']
L_acc_ap=df['L_acc']
R_gyr_ml=df['R_gyr']
L_gyr_ml=df['L_gyr']

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
cor=signal.correlate(sensor_data_f[0],sensor_data_f[1], mode="full")
lags = signal.correlation_lags(sensor_data_f[0].size, sensor_data_f[1].size, mode="full")
lag = lags[np.argmax(cor)]

#Synchronizing both R and L together after correction for the delay
t_R_ml=sensor_data_f.T[round(abs(lag)):-1,0]
t_L_ml=sensor_data_f.T[0:-round(abs(lag))-1,1]

S_sync=np.array([t_R_ml,t_L_ml])
y_r=np.array([],dtype=float)
x_r=np.arange(0,len(t_R_ml)-400,200)
for w in x_r:
    y_r=np.append(y_r,abs(np.corrcoef(t_R_ml[w:w+200],t_L_ml[w:w+200])[0,1]))
    
  
yy_r=np.array([],dtype=float)
xx_r=np.arange(0,len(sensor_data_f.T)-400,200)
for w in xx_r:
    cor=signal.correlate(sensor_data_f.T[w:w+200,0],sensor_data_f.T[w:w+200,1], mode="full")
    lags = signal.correlation_lags(sensor_data_f.T[w:w+200,0].size, sensor_data_f.T[w:w+200,1].size, mode="full")
    lag = lags[np.argmax(cor)]
    s1=sensor_data_f.T[w:w+200,0]
    s2=sensor_data_f.T[w:w+200,1]
    s1_L=s1[round(abs(lag)):-1]
    s2_L=s2[0:-round(abs(lag))-1]
    yy_r=np.append(yy_r,abs(np.corrcoef(s1_L,s2_L)[0,1]))
    # yy_r=np.append(yy_r,abs(np.corrcoef(sensor_data_f.T[w:w+400,0],sensor_data_f.T[w:w+400,1])[0,1]))
    
plt.plot(x_r,y_r,'b',xx_r,yy_r,'r')
plt.show()

plt.plot(sensor_data_f.T[0:5000,:])
plt.show()

plt.plot(S_sync.T[0:5000,:])
plt.show()