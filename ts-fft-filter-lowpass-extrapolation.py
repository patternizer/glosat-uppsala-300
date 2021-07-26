#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: ts-fft-filter-lowpass-extrapolation.py
#------------------------------------------------------------------------------
# Version 0.1
# 25 July, 2021
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
# michael DOT a DOT taylor AT uea DOT ac DOT uk
#------------------------------------------------------------------------------

import numpy as np
from numpy import fft
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

#-----------------------------------------------------------------------------
# SETTINGS
#-----------------------------------------------------------------------------

fontsize = 16
baseline_start = 1981
baseline_end = 2010
#stationcode = '071560'
#stationcode = '744920'
stationcode = '037401'
external_timeseries = False

n_harmonics = 20 # number of harmonics in FFT predictor model

#-----------------------------------------------------------------------------
# METHODS
#-----------------------------------------------------------------------------

def fourierExtrapolation(y, n_predict, n_harmonics):
    
    # https://mran.microsoft.com/posts/timeseries
    # https://stackoverflow.com/questions/4479463/using-fourier-analysis-for-time-series-prediction
    
    n = y.size
    t = np.arange(0, n)
    p = np.polyfit(t, y, 1)         # find linear trend in y       
    y_detrend = y - p[0] * t        # detrended y
    yf = fft.fft(y_detrend)         # detrended y in frequency domain

    h = np.sort(yf)[-n_harmonics]
    yf=[ yf[i] if np.absolute(yf[i]) >= h else 0 for i in range(len(yf)) ]

    freq = fft.fftfreq(n)           # frequencies         
    indexes = list(range(n))
    indexes.sort(key=lambda i: np.absolute(freq[i]) )

    t = np.arange(0, n + n_predict)
    reconstruction = np.zeros(t.size)
    for i in indexes[:1 + n_harmonics * 2]:

        amplitude = np.absolute( yf[i] ) / n    # amplitude
        phase = np.angle(yf[i])                 # phase
        reconstruction += amplitude * np.cos(2 * np.pi * freq[i] * t + phase)

    return reconstruction + p[0] * t

if __name__ == "__main__":
 
    if external_timeseries == True:
                
        #-----------------------------------------------------------------------------
        # GENERATE: signal from sum of sinusoids
        #-----------------------------------------------------------------------------
        
        f = np.array([ 1,4,6,9 ])               # frequencies (kHz)
        a = np.array([ 3,1,0.5,0.5 ])           # coefficients
        M = 8000                                # signal length 
        Fs = 2000                               # sampling rate
        Ts = 1.0/Fs                             # Sampling interval                 
        n = np.arange(M)
        y = np.zeros(M) 
        for fk, ak in zip(f,a): 
            x = fk*Ts*np.arange(M)
            y += ak*np.sin(2*np.pi*x) 
               
        y = y + 0.001 * n 
            
#       y = np.array([  ])

        n_predict = int( len(y)/2 )
        t = np.arange(len(y))
        t_extrapolation = np.arange( len(y) + n_predict )

    else:
        
        #------------------------------------------------------------------------------
        # LOAD: a station anomaly timeseries
        #------------------------------------------------------------------------------
        
        df_temp = pd.read_pickle('DATA/df_temp.pkl', compression='bz2')    
        da = df_temp[df_temp['stationcode'] == stationcode]
        stationname = da['stationname'].iloc[0]
        da = da[da.year >= 1678].reset_index(drop=True)
        db = da.copy()
        for j in range(1,13):
            baseline = np.nanmean(np.array(da[(da['year'] >= 1981) & (da['year'] <= 2010)].iloc[:,j]).ravel())
            db.loc[da.index.tolist(), str(j)] = da[str(j)]-baseline                        
        ts = np.array(db.groupby('year').mean().iloc[:,0:12]).ravel()
        t = pd.date_range(start=str(db.year.iloc[0]), periods=len(ts), freq='MS')
        df = pd.DataFrame({'Tg':ts}, index=t)   
        
        # RESAMPLE: to yearly
        
        df_xr = df.to_xarray()    
        df_xr_resampled = df_xr.Tg.resample(index='AS').mean().to_dataset()        
        y = df_xr_resampled.Tg.values
        t = df_xr_resampled.index.values
    
        t_extrapolation = pd.date_range(start=t[0], end=pd.to_datetime('2100-01-01'), freq='AS')
        n_predict = len(t_extrapolation) - len(y)

    # CALL: FFT predictor
    
    extrapolation = fourierExtrapolation(y, n_predict, n_harmonics)

    #------------------------------------------------------------------------------
    # PLOT: timeseries + FFT low pass + extrapolation
    #------------------------------------------------------------------------------
    
    fig, ax = plt.subplots(figsize=(15,10))
    plt.plot(t, y, ls='-', lw=3, color='red', alpha=0.5, label='Observations (yearly average)')
    plt.plot(t, extrapolation[0:len(y)], ls='-', lw=5, color='lime', label='FFT trend fit: N(harmonics)=' + str(n_harmonics))
    plt.plot(t_extrapolation[len(y):], extrapolation[len(y):], ls='--', lw=3, color='teal', label='FFT extrapolation')
    plt.tick_params(labelsize=fontsize)    
    plt.xlabel('Time', fontsize=fontsize)
    if external_timeseries == True:
        plt.ylabel('Magnitude', fontsize=fontsize)
        plt.title('Timeseries and FFT prediction', fontsize=fontsize)
    else:
        plt.ylabel('Temperature anomaly (from ' + str(baseline_start) + '-' + str(baseline_end) + r'), $^{\circ}$C', fontsize=fontsize)
        plt.title(stationname + ' (' + stationcode + '): observations and FFT prediction to 2100 ', fontsize=fontsize)
    plt.legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)   
    fig.tight_layout()
    plt.savefig('ts-fft-filter-signal-extrpolation.png', dpi=300)
    plt.close('all')
    
#-----------------------------------------------------------------------------
print('** END')

