# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 14:14:36 2021

@author: andre
"""
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
data = pd.read_csv("C:/Users/andre/OneDrive/Documents/Construction_Time_Series_Data_V2.csv")

print(data)

tcon = (data['Total Construction'])
prcon = (data['Private Construction'])
pbcon = (data['Public Construction'])

tconr = tcon.rolling(5).mean().to_list()
prconr = prcon.rolling(5).mean().to_list()
pbconr = pbcon.rolling(5).mean().to_list()


plt.figure(figsize=(10,10))

plt.plot(tcon, c='k', label='Total Construction')
plt.plot(prcon, c='b', label='Private Construction')
plt.plot(pbcon, c='r', label='Public Construction')
plt.plot(tconr, c='g', label='Total Construction Rolling Avergage')
plt.plot(prconr, c='y', label='Private Construction Rolling Average')
plt.plot(pbconr, c='c', label='Public Construction Rolling Average')


plt.ylabel('Number of New Consruction Projects', fontsize=16)

plt.xlabel('Month', fontsize=16)

plt.legend(loc='upper right')
plt.title('Amount of Construction vs. Moving Average of Construction', fontsize=16)

plt.xticks(ticks = len)
#plt.show()

def autocorr(x, t=1):
    return np.corrcoef(np.array([x[:-t], x[t:]]))













