# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 14:14:36 2021

@author: andre
"""
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
data = pd.read_csv("C:/Users/andre/OneDrive/Documents/Construction_Time_Series_Data_V2.csv")

tcon = (data['Total Construction'])
prcon = (data['Private Construction'])
pbcon = (data['Public Construction'])

tconr = tcon.rolling(5).mean().to_list()
prconr = prcon.rolling(5).mean().to_list()
pbconr = pbcon.rolling(5).mean().to_list()

fig,ax = plt.subplots(2,1, sharex=True)
ax[0].plot(tcon, c='k', label='Total Construction')
ax[0].plot(prcon, c='b', label='Private Construction')
ax[0].plot(pbcon, c='r', label='Public Construction')
ax[0].plot(tconr, c='g', label='Total Construction Rolling Avergage')
ax[0].plot(prconr, c='y', label='Private Construction Rolling Average')
ax[0].plot(pbconr, c='c', label='Public Construction Rolling Average')

ax[0].set_ylabel('Number of New Consruction Projects')

ax[0].set_xlabel('Date')

ax[0].legend(loc='upper right')
ax[0].set_title('Construction vs. Moving Average')
#ax[1].plot(x[:-offset],remainder,c='k')

fig.set_size_inches(15,20)
#plt.show()

def autocorr(x, t=1):
    return np.corrcoef(np.array([x[:-t], x[t:]]))













