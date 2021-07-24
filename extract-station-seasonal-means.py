#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: extract-station-sseasonal-means.py
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
plot_smooth = True
plot_seasonal = True

nsmooth = 60
nfft = 16                   # power of 2
#station_code = '037401'    # CET
#station_code = '103810'    # Berlin-Dahlem
#station_code = '744920'    # BHO
station_code = '071560'     # Paris/Montsouris

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

#def cru_filter(x):  
#    return x_filtered

#------------------------------------------------------------------------------    
# LOAD: GloSAT temperature archive: CRUTEM5.0.1.0
#------------------------------------------------------------------------------

if load_glosat == True:
            
    print('loading temperatures ...')
        
    df_temp = pd.read_pickle('DATA/df_temp.pkl', compression='bz2')    
    da = df_temp[df_temp['stationcode']==station_code]
    da = da[da.year >= 1678].reset_index(drop=True)
    db = da.copy()

    if use_anomalies == True:                        
        for j in range(1,13):
            baseline = np.nanmean(np.array(da[(da['year'] >= baseline_start) & (da['year'] <= baseline_end)].iloc[:,j]).ravel())
            db.loc[da.index.tolist(), str(j)] = da[str(j)]-baseline    
                        
    ts = np.array(db.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(db.year.iloc[0]), periods=len(ts), freq='MS')
    df = pd.DataFrame({'Tg':ts}, index=t)     

    # CALCULATE: moving average

    df['MA'] = df['Tg'].rolling(nsmooth, center=True).mean()
    mask = np.isfinite(df['MA'])

    # CALCULATE: LOESS fit
    
    lowess = sm.nonparametric.lowess(df['MA'].values[mask], df['MA'].index[mask], frac=0.1)       
    da = pd.DataFrame({'LOESS':lowess[:,1]}, index=df['MA'].index[mask])
    df['LOESS'] = da                     

    # CALCULATE: FFT low pass

    da = pd.DataFrame({'FFT':smooth_fft(df['MA'].values[mask], nfft)}, index=df['MA'].index[mask])
    df['FFT'] = da[(da.index>da.index[int(nfft/2)]) & (da.index<=da.index[-int(nfft/2)])] # edge effect truncation
    
#------------------------------------------------------------------------------
# PLOTS
#------------------------------------------------------------------------------
    
if plot_smooth == True:
        
    print('plotting filtered timeseries ... ')   
                            
    figstr = station_code + '-fft-smooth.png'
    titlestr = 'GloSAT.p03: Station (' + station_code + ') monthly $T_g$'
               
    fig, ax = plt.subplots(figsize=(15,10))    
    plt.plot(df.index, df['MA'], ls='-', lw=3, color='red', alpha=0.2, zorder=1, label=r'Station (' + station_code + r'): $T_{g}$ 5yr MA')
    plt.plot(df.index, df['FFT'], ls='-', lw=2, color='red', alpha=1, zorder=1, label=r'Station (' + station_code + r': $T_{g}$ 5yr MA (FFT low pass)')
    plt.plot(df.index, df['LOESS'], ls='--', lw=1, color='black', alpha=1, zorder=1, label=r'Station (' + station_code + r'): $T_{g}$ 5yr MA (LOESS, $\alpha$=0.1)')
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

    df_xr = df.to_xarray()    
    df_xr_resampled = df_xr.MA.resample(index='AS').mean().to_dataset()
    df_yearly = pd.DataFrame({'MA':df_xr_resampled.MA.values}, index = df_xr_resampled.index.values)
  
    # RESAMPLE: seasonal weighted means
    
#    month_length = df_uppsala_xr.index.dt.days_in_month
#    weights = month_length.groupby('index.season') / month_length.groupby('index.season').sum()
#    df_seasonal_xr = (df_uppsala_xr * weights).groupby('index.season').sum(dim='index')    
    
    DJF = ( df[df.index.month==12]['Tg'].values + df[df.index.month==1]['Tg'].values + df[df.index.month==2]['Tg'].values ) / 3
    MAM = ( df[df.index.month==3]['Tg'].values + df[df.index.month==4]['Tg'].values + df[df.index.month==5]['Tg'].values ) / 3
    JJA = ( df[df.index.month==6]['Tg'].values + df[df.index.month==7]['Tg'].values + df[df.index.month==8]['Tg'].values ) / 3
    SON = ( df[df.index.month==9]['Tg'].values + df[df.index.month==10]['Tg'].values + df[df.index.month==11]['Tg'].values ) / 3
    df_seasonal = pd.DataFrame({'DJF':DJF, 'MAM':MAM, 'JJA':JJA, 'SON':SON}, index = df_yearly.index.values)
       
    df_seasonal_ma = df_seasonal.rolling(14, center=True).mean()
    mask = np.isfinite(df_seasonal_ma)

    dates = pd.date_range(start='1678-01-01', end='2021-12-01', freq='MS')
    df_seasonal_fft = pd.DataFrame(index=dates)
    df_seasonal_fft['DJF'] = pd.DataFrame({'DJF':smooth_fft(df_seasonal_ma['DJF'].values[mask['DJF']], nfft)}, index=df_seasonal_ma['DJF'].index[mask['DJF']])
    df_seasonal_fft['MAM'] = pd.DataFrame({'DJF':smooth_fft(df_seasonal_ma['MAM'].values[mask['MAM']], nfft)}, index=df_seasonal_ma['MAM'].index[mask['MAM']])
    df_seasonal_fft['JJA'] = pd.DataFrame({'DJF':smooth_fft(df_seasonal_ma['JJA'].values[mask['JJA']], nfft)}, index=df_seasonal_ma['JJA'].index[mask['JJA']])
    df_seasonal_fft['SON'] = pd.DataFrame({'DJF':smooth_fft(df_seasonal_ma['SON'].values[mask['SON']], nfft)}, index=df_seasonal_ma['SON'].index[mask['SON']])

    df_seasonal_fft = df_seasonal_fft[(df_seasonal_fft.index>df_seasonal_fft.index[int(nfft/2)]) & (df_seasonal_fft.index<=df_seasonal_fft.index[-int(nfft/2)])] # edge effect truncation            
    mask = np.isfinite(df_seasonal_fft)

    figstr = station_code + '-seasonal.png'
    titlestr = 'GloSAT.p03: Station (' + station_code + ') seasonal $T_g$: 5yr MA (FFT low pass)'
               
    fig, ax = plt.subplots(figsize=(15,10))    
    plt.plot(df_seasonal_fft.index[mask['DJF']], df_seasonal_fft['DJF'][mask['DJF']], ls='-', lw=3, color='blue', alpha=1, zorder=1, label=r'Winter')
    plt.plot(df_seasonal_fft.index[mask['MAM']], df_seasonal_fft['MAM'][mask['MAM']], ls='-', lw=3, color='red', alpha=1, zorder=1, label=r'Spring')
    plt.plot(df_seasonal_fft.index[mask['JJA']], df_seasonal_fft['JJA'][mask['JJA']], ls='-', lw=3, color='purple', alpha=1, zorder=1, label=r'Summer')
    plt.plot(df_seasonal_fft.index[mask['SON']], df_seasonal_fft['SON'][mask['SON']], ls='-', lw=3, color='green', alpha=1, zorder=1, label=r'Autumn')
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
