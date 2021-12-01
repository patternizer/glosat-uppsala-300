#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: uppsala-stockholm-comparison.py
#------------------------------------------------------------------------------
# Version 0.1
# 20 July, 2021
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
# michael DOT a DOT taylor AT uea DOT ac DOT uk
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# IMPORT PYTHON LIBRARIES
#------------------------------------------------------------------------------
# Dataframe libraries:
import numpy as np
import numpy.ma as ma
import itertools
import pandas as pd
import xarray as xr
import pickle
from datetime import datetime
import nc_time_axis
import cftime

# Plotting libraries:
import matplotlib
import matplotlib.pyplot as plt; plt.close('all')
import matplotlib.cm as cm
from matplotlib import colors as mcol
from matplotlib.cm import ScalarMappable
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter
from matplotlib.collections import PolyCollection
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import cmocean
import seaborn as sns; sns.set()

# OS libraries:
import os
import os.path
from pathlib import Path
import sys
import subprocess
from subprocess import Popen
import time

# Stats libraries:
import random
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess

# Maths libraries
import scipy
from scipy import fftpack
import statsmodels.api as sm

# Datetime libraries
import cftime
import calendar 
# print(calendar.calendar(2020))
# print(calendar.month(2020,2))
# calendar.isleap(2020)
from datetime import date, time, datetime, timedelta
#today = datetime.now()
#tomorrow = today + pd.to_timedelta(1,unit='D')
#tomorrow = today + timedelta(days=1)
#birthday = datetime(1970,11,1,0,0,0).strftime('%Y-%m-%d %H:%M')
#print('Week:',today.isocalendar()[1])

# Silence library version notifications
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# SETTINGS: 
#------------------------------------------------------------------------------

fontsize = 16
use_fahrenheit = False
baseline_start = 1981
baseline_end = 2010

if use_fahrenheit: 
    temperature_unit = 'F'
else:
    temperature_unit = 'C'

load_glosat = True
use_anomalies = True
plot_differences = True
plot_smooth = True
plot_seasonal = True

nsmooth = 60
nfft = 32                   # power of 2

#------------------------------------------------------------------------------
# METHODS: 
#------------------------------------------------------------------------------

def linear_regression_ols(x,y):

    regr = linear_model.LinearRegression()
    # regr = TheilSenRegressor(random_state=42)
    # regr = RANSACRegressor(random_state=42)

    X = x[:, np.newaxis]    
    # X = x.values.reshape(len(x),1)
    t = np.linspace(X.min(),X.max(),len(X)) # dummy var spanning [xmin,xmax]        
    regr.fit(X, y)
    ypred = regr.predict(t.reshape(-1, 1))
    slope = regr.coef_
    intercept = regr.intercept_
    mse = mean_squared_error(y,ypred)
    r2 = r2_score(y,ypred) 
    
    return t, ypred, slope, intercept, mse, r2

def fahrenheit_to_centigrade(x):
    y = (5.0/9.0) * (x - 32.0)
    return y

def centigrade_to_fahrenheit(x):
    y = (x * (9.0/5.0)) + 32.0
    return y

def is_leap_and_29Feb(s):
    return (s.index.year % 4 == 0) & ((s.index.year % 100 != 0) | (s.index.year % 400 == 0)) & (s.index.month == 2) & (s.index.day == 29)

def convert_datetime_to_year_decimal(df, yearstr):

    if yearstr == 'datetime':
        t_monthly_xr = xr.cftime_range(start=str(df.index.year[0]), periods=len(df), freq='MS', calendar='all_leap')
    else:
        t_monthly_xr = xr.cftime_range(start=str(df[yearstr].iloc[0]), periods=len(df)*12, freq='MS', calendar='all_leap')
    year = [t_monthly_xr[i].year for i in range(len(t_monthly_xr))]
    year_frac = []
    for i in range(len(t_monthly_xr)):
        if i%12 == 0:
            istart = i
            iend = istart+11                  
            frac = np.cumsum([t_monthly_xr[istart+j].day for j in range(12)])               
            year_frac += list(frac/frac[-1])            
        else:                
            i += 1
    year_decimal = [float(year[i])+year_frac[i] for i in range(len(year))]    
    return year_decimal
    
def smooth_fft(x, span):  
    
    # TO DO: make this more robust with a Hanning window
    
    y = x - np.nanmean(x)
    w = scipy.fftpack.rfft(y) 
    spectrum = w**2 
    cutoff_idx = spectrum < (spectrum.max() * (1 - np.exp(-span / 2000)))
#   cutoff_idx = spectrum < spectrum.max() * 0.1
    w[cutoff_idx] = 0    
    y_filtered = fftpack.irfft(w)
    x_filtered = y_filtered + np.nanmean(x)
    return x_filtered

#------------------------------------------------------------------------------    
# LOAD: GloSAT temperature archive: CRUTEM5.0.1.0
#------------------------------------------------------------------------------

if load_glosat == True:
            
    print('loading temperatures ...')
        
    df_temp = pd.read_pickle('DATA/df_temp.pkl', compression='bz2')    
    
    stationcode_uppsala = '024581'
    stationcode_stockholm = '024851'

    da_uppsala = df_temp[df_temp['stationcode']==stationcode_uppsala]                       
    da_stockholm = df_temp[df_temp['stationcode']==stationcode_stockholm]      
    db_uppsala = da_uppsala.copy()
    db_stockholm = da_stockholm.copy()

    if use_anomalies == True:                        
        for j in range(1,13):
            baseline_uppsala = np.nanmean(np.array(da_uppsala[(da_uppsala['year'] >= baseline_start) & (da_uppsala['year'] <= baseline_end)].iloc[:,j]).ravel())
            baseline_stockholm = np.nanmean(np.array(da_stockholm[(da_stockholm['year'] >= baseline_start) & (da_stockholm['year'] <= baseline_end)].iloc[:,j]).ravel())
            db_uppsala.loc[da_uppsala.index.tolist(), str(j)] = da_uppsala[str(j)]-baseline_uppsala                  
            db_stockholm.loc[db_stockholm.index.tolist(), str(j)] = db_stockholm[str(j)]-baseline_stockholm                  

    ts = np.array(db_uppsala.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(db_uppsala.year.iloc[0]), periods=len(ts), freq='MS')
    df_uppsala = pd.DataFrame({'Tg':ts}, index=t) 
    
    ts = np.array(db_stockholm.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(db_stockholm.year.iloc[0]), periods=len(ts), freq='MS')
    df_stockholm = pd.DataFrame({'Tg':ts}, index=t) 

    # CALCULATE: moving average    

    df_uppsala['MA'] = df_uppsala['Tg'].rolling(nsmooth, center=True).mean()
    df_stockholm['MA'] = df_stockholm['Tg'].rolling(nsmooth, center=True).mean()
    mask_uppsala = np.isfinite(df_uppsala['MA'])
    mask_stockholm = np.isfinite(df_stockholm['MA'])

    # CALCULATE: LOESS fit
    
    lowess = sm.nonparametric.lowess(df_uppsala['MA'].values[mask_uppsala], df_uppsala['MA'].index[mask_uppsala], frac=0.1)       
    da = pd.DataFrame({'LOESS':lowess[:,1]}, index=df_uppsala['MA'].index[mask_uppsala])
    df_uppsala['LOESS'] = da                     
    lowess = sm.nonparametric.lowess(df_stockholm['MA'].values[mask_stockholm], df_stockholm['MA'].index[mask_stockholm], frac=0.1)       
    da = pd.DataFrame({'LOESS':lowess[:,1]}, index=df_stockholm['MA'].index[mask_stockholm])
    df_stockholm['LOESS'] = da       

    # CALCULATE: FFT low pass

    da = pd.DataFrame({'FFT':smooth_fft(df_uppsala['MA'].values[mask_uppsala], nfft)}, index=df_uppsala['MA'].index[mask_uppsala])
    df_uppsala['FFT'] = da[(da.index>da.index[int(nfft/2)]) & (da.index<=da.index[-int(nfft/2)])] # edge effect truncation
    da = pd.DataFrame({'FFT':smooth_fft(df_stockholm['MA'].values[mask_stockholm], 12)}, index=df_stockholm['MA'].index[mask_stockholm])
    df_stockholm['FFT'] = da[(da.index>da.index[int(nfft/2)]) & (da.index<=da.index[-int(nfft/2)])] # edge effect truncation
        
#------------------------------------------------------------------------------
# PLOTS
#------------------------------------------------------------------------------

if plot_differences == True:
            
    # PLOT: Uppsala vs Stockholm and differences: Tg (monthly)
    
    print('plotting timeseries and differences: monthly Tg ...')
        
    figstr = 'uppsala-and-stockholm.png'
    titlestr = 'GloSAT.p03: Uppsala (024581) versus Stockholm (024851) monthly $T_g$'
    
    mask = np.isfinite(df_uppsala.Tg) & np.isfinite(df_stockholm.Tg)
    
    fig, axs = plt.subplots(2,1, figsize=(15,10))
    sns.lineplot(x=df_uppsala.index, y='Tg', data=df_uppsala, ax=axs[0], marker='o', color='red', alpha=1.0, label='Uppsala (024581) $T_{g}$')
    sns.lineplot(x=df_stockholm.index, y='Tg', data=df_stockholm, ax=axs[0], marker='.', color='blue', alpha=1.0, label='Stockholm (024851) $T_{g}$')    
    sns.lineplot(x=df_uppsala.index[mask], y=df_uppsala['Tg'][mask]-df_stockholm['Tg'][mask], ax=axs[1], color='teal')    
    plt.axhline(y=0.0, ls='dashed', lw=1, color='black')
    axs[0].legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)       
    axs[0].set_xlabel('', fontsize=fontsize)
    if use_anomalies == True:
        axs[0].set_ylabel(r'2m Temperature anomaly (from ' + str(baseline_start) + '-' + str(baseline_end) + r'), $^{\circ}$' + temperature_unit, fontsize=fontsize)
    else:
        axs[0].set_ylabel(r'2m Temperature, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    axs[0].set_title(titlestr, fontsize=fontsize)
    axs[0].tick_params(labelsize=fontsize)   
    axs[1].sharex(axs[0]) 
    axs[1].tick_params(labelsize=fontsize)    
    axs[1].set_xlabel('Year', fontsize=fontsize)
    axs[1].set_ylabel(r'Uppsala $T_{g}$ - Stockholm $T_{g}$, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')

#==============================================================================
    
if plot_smooth == True:
        
    print('plotting filtered timeseries ... ')   
                                          
    figstr = 'uppsala-and-stockholm-fft-smooth.png'
    titlestr = 'GloSAT.p03: Uppsala (024581) and Stockholm (024851) monthly $T_g$'
               
    fig, ax = plt.subplots(figsize=(15,10))    
    plt.plot(df_uppsala.index, df_uppsala['MA'], ls='-', lw=3, color='red', alpha=0.2, zorder=1, label=r'Uppsala (024581): $T_{g}$ 5yr MA')
    plt.plot(df_uppsala.index, df_uppsala['FFT'], ls='-', lw=2, color='red', alpha=1, zorder=1, label=r'Uppsala (024581): $T_{g}$ 5yr MA (FFT low pass)')
    plt.plot(df_uppsala.index, df_uppsala['LOESS'], ls='--', lw=1, color='red', alpha=1, zorder=1, label=r'Uppsala (024581): $T_{g}$ 5yr MA (LOESS, $\alpha$=0.1)')
    plt.plot(df_stockholm.index, df_stockholm['MA'], ls='-', lw=3, color='blue', alpha=0.2, zorder=1, label=r'Stockholm (024851): $T_{g}$ 5yr MA')
    plt.plot(df_stockholm.index, df_stockholm['FFT'], ls='-', lw=2, color='blue', alpha=1, zorder=1, label=r'Stockholm (024851): $T_{g}$ 5yr MA (FFT low pass)')
    plt.plot(df_stockholm.index, df_stockholm['LOESS'], ls='--', lw=1, color='blue', alpha=1, zorder=1, label=r'Stockholm (024851): $T_{g}$ 5yr MA (LOESS, $\alpha$=0.1)')
    plt.xlabel('Year', fontsize=fontsize)
    if use_anomalies == True:
        plt.ylabel(r'2m Temperature anomaly (from ' + str(baseline_start) + '-' + str(baseline_end) + r'), $^{\circ}$' + temperature_unit, fontsize=fontsize)
    else:
        plt.ylabel(r'2m Temperature, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    plt.title(titlestr, fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)    
    plt.legend(loc='upper left', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')

#==============================================================================

if plot_seasonal == True:
        
    print('plotting seasonal timeseries ... ')   

    # RESAMPLE: to yearly using xarray

    df_uppsala_xr = df_uppsala.to_xarray()    
    df_uppsala_xr_resampled = df_uppsala_xr.MA.resample(index='AS').mean().to_dataset()
    df_uppsala_yearly = pd.DataFrame({'MA':df_uppsala_xr_resampled.MA.values}, index = df_uppsala_xr_resampled.index.values)
  
    # RESAMPLE: seasonal weighted means
    
#    month_length = df_uppsala_xr.index.dt.days_in_month
#    weights = month_length.groupby('index.season') / month_length.groupby('index.season').sum()
#    df_seasonal_xr = (df_uppsala_xr * weights).groupby('index.season').sum(dim='index')    
    
    DJF = ( df_uppsala[df_uppsala.index.month==12]['Tg'].values + df_uppsala[df_uppsala.index.month==1]['Tg'].values + df_uppsala[df_uppsala.index.month==2]['Tg'].values ) / 3
    MAM = ( df_uppsala[df_uppsala.index.month==3]['Tg'].values + df_uppsala[df_uppsala.index.month==4]['Tg'].values + df_uppsala[df_uppsala.index.month==5]['Tg'].values ) / 3
    JJA = ( df_uppsala[df_uppsala.index.month==6]['Tg'].values + df_uppsala[df_uppsala.index.month==7]['Tg'].values + df_uppsala[df_uppsala.index.month==8]['Tg'].values ) / 3
    SON = ( df_uppsala[df_uppsala.index.month==9]['Tg'].values + df_uppsala[df_uppsala.index.month==10]['Tg'].values + df_uppsala[df_uppsala.index.month==11]['Tg'].values ) / 3
    df_uppsala_seasonal = pd.DataFrame({'DJF':DJF, 'MAM':MAM, 'JJA':JJA, 'SON':SON}, index = df_uppsala_yearly.index.values)
       
    df_uppsala_seasonal_ma = df_uppsala_seasonal.rolling(14, center=True).mean()
    mask = np.isfinite(df_uppsala_seasonal_ma)

    dates = pd.date_range(start='1678-01-01', end='2021-12-01', freq='MS')
    df_uppsala_seasonal_fft = pd.DataFrame(index=dates)
    df_uppsala_seasonal_fft['DJF'] = pd.DataFrame({'DJF':smooth_fft(df_uppsala_seasonal_ma['DJF'].values[mask['DJF']], nfft)}, index=df_uppsala_seasonal_ma['DJF'].index[mask['DJF']])
    df_uppsala_seasonal_fft['MAM'] = pd.DataFrame({'DJF':smooth_fft(df_uppsala_seasonal_ma['MAM'].values[mask['MAM']], nfft)}, index=df_uppsala_seasonal_ma['MAM'].index[mask['MAM']])
    df_uppsala_seasonal_fft['JJA'] = pd.DataFrame({'DJF':smooth_fft(df_uppsala_seasonal_ma['JJA'].values[mask['JJA']], nfft)}, index=df_uppsala_seasonal_ma['JJA'].index[mask['JJA']])
    df_uppsala_seasonal_fft['SON'] = pd.DataFrame({'DJF':smooth_fft(df_uppsala_seasonal_ma['SON'].values[mask['SON']], nfft)}, index=df_uppsala_seasonal_ma['SON'].index[mask['SON']])

    df_uppsala_seasonal_fft = df_uppsala_seasonal_fft[(df_uppsala_seasonal_fft.index>df_uppsala_seasonal_fft.index[int(nfft/2)]) & (df_uppsala_seasonal_fft.index<=df_uppsala_seasonal_fft.index[-int(nfft/2)])] # edge effect truncation    
    mask = np.isfinite(df_uppsala_seasonal_fft)

    figstr = 'uppsala-seasonal.png'
    titlestr = 'GloSAT.p03: Uppsala (024581) seasonal $T_g$: 14yr MA (FFT low pass)'
               
    fig, ax = plt.subplots(figsize=(15,10))    
    plt.plot(df_uppsala_seasonal_fft.index[mask['DJF']], df_uppsala_seasonal_fft['DJF'][mask['DJF']], ls='-', lw=3, color='blue', alpha=1, zorder=1, label=r'Winter')
    plt.plot(df_uppsala_seasonal_fft.index[mask['MAM']], df_uppsala_seasonal_fft['MAM'][mask['MAM']], ls='-', lw=3, color='red', alpha=1, zorder=1, label=r'Spring')
    plt.plot(df_uppsala_seasonal_fft.index[mask['JJA']], df_uppsala_seasonal_fft['JJA'][mask['JJA']], ls='-', lw=3, color='purple', alpha=1, zorder=1, label=r'Summer')
    plt.plot(df_uppsala_seasonal_fft.index[mask['SON']], df_uppsala_seasonal_fft['SON'][mask['SON']], ls='-', lw=3, color='green', alpha=1, zorder=1, label=r'Autumn')
    plt.xlabel('Year', fontsize=fontsize)
    ax.set_xlim(pd.to_datetime('1720-01-01'),pd.to_datetime('2021-01-01'))
    ax.set_ylim(-4,2)         
    if use_anomalies == True:
        plt.ylabel(r'2m Temperature anomaly (from ' + str(baseline_start) + '-' + str(baseline_end) + r'), $^{\circ}$' + temperature_unit, fontsize=fontsize)
    else:
        plt.ylabel(r'2m Temperature, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    plt.title(titlestr, fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)    
    plt.legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')

#==============================================================================
                   
    df_stockholm_xr = df_stockholm.to_xarray()    
    df_stockholm_xr_resampled = df_stockholm_xr.MA.resample(index='AS').mean().to_dataset()
    df_stockholm_yearly = pd.DataFrame({'MA':df_stockholm_xr_resampled.MA.values}, index = df_stockholm_xr_resampled.index.values)
  
    # RESAMPLE: seasonal weighted means
        
    DJF = ( df_stockholm[df_stockholm.index.month==12]['Tg'].values + df_stockholm[df_stockholm.index.month==1]['Tg'].values + df_stockholm[df_stockholm.index.month==2]['Tg'].values ) / 3
    MAM = ( df_stockholm[df_stockholm.index.month==3]['Tg'].values + df_stockholm[df_stockholm.index.month==4]['Tg'].values + df_stockholm[df_stockholm.index.month==5]['Tg'].values ) / 3
    JJA = ( df_stockholm[df_stockholm.index.month==6]['Tg'].values + df_stockholm[df_stockholm.index.month==7]['Tg'].values + df_stockholm[df_stockholm.index.month==8]['Tg'].values ) / 3
    SON = ( df_stockholm[df_stockholm.index.month==9]['Tg'].values + df_stockholm[df_stockholm.index.month==10]['Tg'].values + df_stockholm[df_stockholm.index.month==11]['Tg'].values ) / 3
    df_stockholm_seasonal = pd.DataFrame({'DJF':DJF, 'MAM':MAM, 'JJA':JJA, 'SON':SON}, index = df_stockholm_yearly.index.values)
       
    df_stockholm_seasonal_ma = df_stockholm_seasonal.rolling(14, center=True).mean()
    mask = np.isfinite(df_stockholm_seasonal_ma)

    dates = pd.date_range(start='1678-01-01', end='2021-12-01', freq='MS')
    df_stockholm_seasonal_fft = pd.DataFrame(index=dates)
    df_stockholm_seasonal_fft['DJF'] = pd.DataFrame({'DJF':smooth_fft(df_stockholm_seasonal_ma['DJF'].values[mask['DJF']], nfft)}, index=df_stockholm_seasonal_ma['DJF'].index[mask['DJF']])
    df_stockholm_seasonal_fft['MAM'] = pd.DataFrame({'DJF':smooth_fft(df_stockholm_seasonal_ma['MAM'].values[mask['MAM']], nfft)}, index=df_stockholm_seasonal_ma['MAM'].index[mask['MAM']])
    df_stockholm_seasonal_fft['JJA'] = pd.DataFrame({'DJF':smooth_fft(df_stockholm_seasonal_ma['JJA'].values[mask['JJA']], nfft)}, index=df_stockholm_seasonal_ma['JJA'].index[mask['JJA']])
    df_stockholm_seasonal_fft['SON'] = pd.DataFrame({'DJF':smooth_fft(df_stockholm_seasonal_ma['SON'].values[mask['SON']], nfft)}, index=df_stockholm_seasonal_ma['SON'].index[mask['SON']])

    df_stockholm_seasonal_fft = df_stockholm_seasonal_fft[(df_stockholm_seasonal_fft.index>df_stockholm_seasonal_fft.index[int(nfft/2)]) & (df_stockholm_seasonal_fft.index<=df_stockholm_seasonal_fft.index[-int(nfft/2)])] # edge effect truncation    
    mask = np.isfinite(df_stockholm_seasonal_fft)

    figstr = 'stockholm-seasonal.png'
    titlestr = 'GloSAT.p03: Stockholm (024851) seasonal $T_g$: 14yr MA (FFT low pass)'
               
    fig, ax = plt.subplots(figsize=(15,10))    
    plt.plot(df_stockholm_seasonal_fft.index[mask['DJF']], df_stockholm_seasonal_fft['DJF'][mask['DJF']], ls='-', lw=3, color='blue', alpha=1, zorder=1, label=r'Winter')
    plt.plot(df_stockholm_seasonal_fft.index[mask['MAM']], df_stockholm_seasonal_fft['MAM'][mask['MAM']], ls='-', lw=3, color='red', alpha=1, zorder=1, label=r'Spring')
    plt.plot(df_stockholm_seasonal_fft.index[mask['JJA']], df_stockholm_seasonal_fft['JJA'][mask['JJA']], ls='-', lw=3, color='purple', alpha=1, zorder=1, label=r'Summer')
    plt.plot(df_stockholm_seasonal_fft.index[mask['SON']], df_stockholm_seasonal_fft['SON'][mask['SON']], ls='-', lw=3, color='green', alpha=1, zorder=1, label=r'Autumn')
    ax.set_xlim(pd.to_datetime('1720-01-01'),pd.to_datetime('2021-01-01'))
    ax.set_ylim(-4,2)             
    plt.xlabel('Year', fontsize=fontsize)
    if use_anomalies == True:
        plt.ylabel(r'2m Temperature anomaly (from ' + str(baseline_start) + '-' + str(baseline_end) + r'), $^{\circ}$' + temperature_unit, fontsize=fontsize)
    else:
        plt.ylabel(r'2m Temperature, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    plt.title(titlestr, fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)    
    plt.legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')    
        
#------------------------------------------------------------------------------
print('** END')
