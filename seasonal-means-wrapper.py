#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: seasonal-means-wrapper.py
#------------------------------------------------------------------------------
# Version 0.2
# 29 November, 2021
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
import netCDF4
from netCDF4 import Dataset, num2date, date2num

# Plotting libraries:
import matplotlib
import matplotlib.pyplot as plt; plt.close('all')
import matplotlib.cm as cm
from matplotlib import colors as mcol
from matplotlib.cm import ScalarMappable
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from matplotlib import rcParams
#import matplotlib.dates as mdates
#import matplotlib.colors as mcolors
#import matplotlib.ticker as mticker
#from matplotlib.ticker import FuncFormatter
#from matplotlib.collections import PolyCollection
#from mpl_toolkits import mplot3d
#from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits.mplot3d import proj3d
#import cmocean
#import seaborn as sns; sns.set()

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
import filter_cru_dft as cru # CRU DFT filter
#import filter_cru_dft as cru_filter # CRU DFT filter
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# SETTINGS: 
#------------------------------------------------------------------------------

fontsize = 16
use_dark_theme = False

baseline_start = 1981
baseline_end = 2010
use_fahrenheit = False
if use_fahrenheit: 
    temperature_unit = 'F'
else:
    temperature_unit = 'C'

use_anomalies = True

nsmooth = 60                 # 5yr MA monthly
nfft = 16                    # power of 2 for the DFT

#station_code = '037401'     # CET
#station_code = '744920'     # BHO
#station_code = '071560'     # Paris/Montsouris
#station_code = '024851'     # Stockholm
station_code = '024581'     # Uppsala 
#station_code = '062600'     # De Bilt
#station_code = '103810'     # Berlin-Dahlem (breakpoint: 1908)
#station_code = '685880'     # Durban/Louis Botha (breakpoint: 1939)
#station_code = '680320'     # Maun (breakpoint: 1925)

#----------------------------------------------------------------------------
# DARK THEME
#----------------------------------------------------------------------------

if use_dark_theme == True:
    
    matplotlib.rcParams['text.usetex'] = False
#    rcParams['font.family'] = 'sans-serif'
#    rcParams['font.sans-serif'] = ['Avant Garde', 'Lucida Grande', 'Verdana', 'DejaVu Sans' ]
    plt.rc('text',color='white')
    plt.rc('lines',color='white')
    plt.rc('patch',edgecolor='white')
    plt.rc('grid',color='lightgray')
    plt.rc('xtick',color='white')
    plt.rc('ytick',color='white')
    plt.rc('axes',labelcolor='white')
    plt.rc('axes',facecolor='black')
    plt.rc('axes',edgecolor='lightgray')
    plt.rc('figure',facecolor='black')
    plt.rc('figure',edgecolor='black')
    plt.rc('savefig',edgecolor='black')
    plt.rc('savefig',facecolor='black')
    
else:

    matplotlib.rcParams['text.usetex'] = False
#    rcParams['font.family'] = 'sans-serif'
#    rcParams['font.sans-serif'] = ['Avant Garde', 'Lucida Grande', 'Verdana', 'DejaVu Sans' ]
    plt.rc('text',color='black')
    plt.rc('lines',color='black')
    plt.rc('patch',edgecolor='black')
    plt.rc('grid',color='lightgray')
    plt.rc('xtick',color='black')
    plt.rc('ytick',color='black')
    plt.rc('axes',labelcolor='black')
    plt.rc('axes',facecolor='white')    
    plt.rc('axes',edgecolor='black')
    plt.rc('figure',facecolor='white')
    plt.rc('figure',edgecolor='white')
    plt.rc('savefig',edgecolor='white')
    plt.rc('savefig',facecolor='white')

# Calculate current time

now = datetime.now()
currentdy = str(now.day).zfill(2)
currentmn = str(now.month).zfill(2)
currentyr = str(now.year)
titletime = str(currentdy) + '/' + currentmn + '/' + currentyr    

#------------------------------------------------------------------------------
# METHODS: 
#------------------------------------------------------------------------------
    
def smooth_fft(x, span):  

    y_lo, y_hi, zvarlo, zvarhi, fc, pctl = cru.cru_filter_dft(x, span)    
    x_filtered = y_lo

    return x_filtered

def merge_fix_cols(df1,df2,var):
    '''
    df1: full time range dataframe (container)
    df2: observation time range
    df_merged: merge of observations into container
    var: 'time' or name of datetime column
    '''
    
    df_merged = pd.merge( df1, df2, how='left', left_on=var, right_on=var)    
#   df_merged = pd.merge( df1, df2, how='left', on=var)    
#   df_merged = df1.merge(df2, how='left', on=var)
    
    for col in df_merged:
        if col.endswith('_y'):
            df_merged.rename(columns = lambda col:col.rstrip('_y'),inplace=True)
        elif col.endswith('_x'):
            to_drop = [col for col in df_merged if col.endswith('_x')]
            df_merged.drop( to_drop, axis=1, inplace=True)
        else:
            pass
    return df_merged
    
#------------------------------------------------------------------------------    
# LOAD: GloSAT temperature archive: CRUTEM5.0.1.0
#------------------------------------------------------------------------------
            
print('loading temperatures ...')
        
df_temp = pd.read_pickle('DATA/df_temp.pkl', compression='bz2')    
df = df_temp[df_temp['stationcode']==station_code]
station_name = df['stationname'].unique()[0]
    
# SET: time axis container
        
# df_yearly = pd.DataFrame({'year': np.arange( 1780, 2021 )}) # 1780-2020 inclusive
df_yearly = pd.DataFrame({'year': np.arange( 1678, 2022 )}) # 1678-2020 inclusive
dt = df_yearly.merge(df, how='left', on='year')
    
# TRIM: to start of Pandas datetime range ( if needed )

dt = dt[dt.year >= 1678].reset_index(drop=True)

# CONVERT: to anomalies ( if use_anomalies = True )

da = dt.copy()
if use_anomalies == True:                        
    for j in range(1,13):
        baseline = np.nanmean(np.array(dt[(dt['year'] >= baseline_start) & (dt['year'] <= baseline_end)].iloc[:,j]).ravel())
        da.loc[dt.index.tolist(), str(j)] = dt[str(j)]-baseline    
                        
# EXTRACT: monthly timeseries

ts_monthly = np.array( da.groupby('year').mean().iloc[:,0:12] ).ravel().astype(float)

# SET: monthly and seasonal time vectors

t_monthly = pd.date_range(start=str(da.year.iloc[0]), periods=len(ts_monthly), freq='MS')
t_seasonal = [ pd.to_datetime( str(da['year'].iloc[i+1])+'-01-01') for i in range(2021-1678) ] # Timestamp('YYYY-01-01 00:00:00')]

df = pd.DataFrame({'Tg':ts_monthly}, index=t_monthly)     

# CALCULATE: moving average

df['MA'] = df['Tg'].rolling(nsmooth, center=True).mean()
mask = np.isfinite(df['MA'])

# RESAMPLE: to yearly using xarray

df_xr = df.to_xarray()    
df_xr_resampled = df_xr.Tg.resample(index='AS').mean().to_dataset()
df_xr_resampled_sd = df_xr.Tg.resample(index='AS').std().to_dataset()
df_yearly = pd.DataFrame({'Tg':df_xr_resampled.Tg.values}, index = df_xr_resampled.index.values)
df_yearly_sd = pd.DataFrame({'Tg':df_xr_resampled_sd.Tg.values}, index = df_xr_resampled_sd.index.values)
   
# CALCULATE: LOESS fit (use Pandas rolling to interpolate and
    
lowess = sm.nonparametric.lowess(df['Tg'].values[mask], df['Tg'].index[mask], frac=0.1)       
da = pd.DataFrame({'LOESS':lowess[:,1]}, index=df['Tg'].index[mask])
df['LOESS'] = da                     

# CALCULATE: FFT low pass

da = pd.DataFrame({'FFT':smooth_fft(df['MA'].values[mask], nfft)}, index=df['MA'].index[mask])
df['FFT'] = da.rolling(nsmooth, center=True).mean()
# df['FFT'] = da[(da.index>da.index[int(nfft/2)]) & (da.index<=da.index[-int(nfft/2)])] # edge effect truncation
    
# RESAMPLE: seasonal weighted means
    
#    month_length = df_uppsala_xr.index.dt.days_in_month
#    weights = month_length.groupby('index.season') / month_length.groupby('index.season').sum()
#    df_seasonal_xr = (df_uppsala_xr * weights).groupby('index.season').sum(dim='index')    
    
# EXTRACT: seasonal components ( D from first year only --> N-1 seasonal estimates )

trim_months = len(t_monthly)%12
da = pd.DataFrame({'Tg':ts_monthly[:-1-trim_months]}, index=t_monthly[:-1-trim_months])         
DJF = ( da[da.index.month==12]['Tg'].values + da[da.index.month==1]['Tg'].values[1:] + da[da.index.month==2]['Tg'].values[1:] ) / 3
MAM = ( da[da.index.month==3]['Tg'].values[1:] + da[da.index.month==4]['Tg'].values[1:] + da[da.index.month==5]['Tg'].values[1:] ) / 3
JJA = ( da[da.index.month==6]['Tg'].values[1:] + da[da.index.month==7]['Tg'].values[1:] + da[da.index.month==8]['Tg'].values[1:] ) / 3
SON = ( da[da.index.month==9]['Tg'].values[1:] + da[da.index.month==10]['Tg'].values[1:] + da[da.index.month==11]['Tg'].values[1:] ) / 3
ONDJFM = ( da[da.index.month==10]['Tg'].values[1:] + da[da.index.month==11]['Tg'].values[1:] + da[da.index.month==12]['Tg'].values + da[da.index.month==1]['Tg'].values[1:] + da[da.index.month==2]['Tg'].values[1:] + da[da.index.month==3]['Tg'].values[1:] ) / 6
AMJJAS = ( da[da.index.month==4]['Tg'].values[1:] + da[da.index.month==5]['Tg'].values[1:] + da[da.index.month==6]['Tg'].values[1:] + da[da.index.month==7]['Tg'].values[1:] + da[da.index.month==8]['Tg'].values[1:] + da[da.index.month==9]['Tg'].values[1:] ) / 6
df_seasonal = pd.DataFrame({'DJF':DJF, 'MAM':MAM, 'JJA':JJA, 'SON':SON, 'ONDJFM':ONDJFM, 'AMJJAS':AMJJAS}, index = t_seasonal)     

mask = np.isfinite(df_seasonal)
df_seasonal_fft = pd.DataFrame(index=df_seasonal.index)
df_seasonal_fft['DJF'] = pd.DataFrame({'DJF':smooth_fft(df_seasonal['DJF'].values[mask['DJF']], nfft)}, index=df_seasonal['DJF'].index[mask['DJF']])
df_seasonal_fft['MAM'] = pd.DataFrame({'DJF':smooth_fft(df_seasonal['MAM'].values[mask['MAM']], nfft)}, index=df_seasonal['MAM'].index[mask['MAM']])
df_seasonal_fft['JJA'] = pd.DataFrame({'DJF':smooth_fft(df_seasonal['JJA'].values[mask['JJA']], nfft)}, index=df_seasonal['JJA'].index[mask['JJA']])
df_seasonal_fft['SON'] = pd.DataFrame({'DJF':smooth_fft(df_seasonal['SON'].values[mask['SON']], nfft)}, index=df_seasonal['SON'].index[mask['SON']])
df_seasonal_fft['ONDJFM'] = pd.DataFrame({'ONDJFM':smooth_fft(df_seasonal['ONDJFM'].values[mask['ONDJFM']], nfft)}, index=df_seasonal['ONDJFM'].index[mask['ONDJFM']])
df_seasonal_fft['AMJJAS'] = pd.DataFrame({'AMJJAS':smooth_fft(df_seasonal['AMJJAS'].values[mask['AMJJAS']], nfft)}, index=df_seasonal['AMJJAS'].index[mask['AMJJAS']])
df_seasonal_fft = df_seasonal_fft[ (df_seasonal_fft.index > df_seasonal_fft.index[int(nfft/2)]) & (df_seasonal_fft.index <= df_seasonal_fft.index[-int(nfft/2)])] # edge effect truncation                   

# SMOOTH: seasonal extracts ( decadal )

#df_seasonal_fft = df_seasonal.copy()[mask].rolling(10, center=True).mean()

#==============================================================================
# PLOTS
#==============================================================================

if use_dark_theme == True:
    default_color = 'white'
else:    
    default_color = 'black'    	
    
#------------------------------------------------------------------------------    
# PLOT: monthly timeseries + smooth + LOESS
#------------------------------------------------------------------------------    
    
print('plotting filtered timeseries ... ')   
                            
figstr = station_code + '-' + 'wrapper' + '-' + 'fft-smooth.png'
titlestr = 'GloSAT.p03: ' + station_name.upper() + ' (' + station_code + ') monthly $T_g$'
               
fig, ax = plt.subplots(figsize=(15,10))    
# plt.plot(df.index, df['Tg'], ls='-', lw=1, color='black', alpha=0.1, zorder=0, label=r'Station (' + station_code + r'): $T_{g}$ monthly')
# plt.plot(df_yearly, ls='-', lw=1, color='black', alpha=0.5, zorder=0, label=r'Station (' + station_code + r'): $T_{g}$ yearly')
# plt.fill_between(df_yearly.index, df_yearly.Tg-df_yearly_sd.Tg, df_yearly.Tg+df_yearly_sd.Tg, ls='-', lw=1, color='black', alpha=0.2, zorder=0, label=r'Yearly $\mu\pm\sigma$')
plt.plot(df.index, df['MA'], ls='-', lw=3, color='red', alpha=0.2, zorder=1, label=r'Station (' + station_code + r'): $T_{g}$ 5yr MA')
plt.plot(df.index, df['FFT'], ls='-', lw=2, color='red', alpha=1, zorder=1, label=r'Station (' + station_code + r': $T_{g}$ 5yr MA (FFT low pass)')
plt.plot(df.index, df['LOESS'], ls='--', lw=2, color=default_color, alpha=1, zorder=1, label=r'Station (' + station_code + r'): $T_{g}$ LOESS ($\alpha$=0.1)')
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
# PLOT: seasonal extracts
#------------------------------------------------------------------------------    

print('plotting seasonal timeseries ... ')   
  
figstr = station_code + '-' + 'wrapper' + '-' + 'seasonal.png'
titlestr = 'GloSAT.p03: ' + station_name.upper() + ' (' + station_code + ') seasonal $T_g$: decadal and FFT low pass fit'
               
fig, ax = plt.subplots(figsize=(15,10))    
plt.plot(df_seasonal.index, df_seasonal['DJF'].rolling(10,center=True).mean(), ls='-', lw=3, color='blue', alpha=0.2, zorder=1)
plt.plot(df_seasonal.index, df_seasonal['MAM'].rolling(10,center=True).mean(), ls='-', lw=3, color='red', alpha=0.2, zorder=1)
plt.plot(df_seasonal.index, df_seasonal['JJA'].rolling(10,center=True).mean(), ls='-', lw=3, color='purple', alpha=0.2, zorder=1)
plt.plot(df_seasonal.index, df_seasonal['SON'].rolling(10,center=True).mean(), ls='-', lw=3, color='green', alpha=0.2, zorder=1)
#plt.plot(df_seasonal.index, df_seasonal['AMJJAS'].rolling(10,center=True).mean(), ls='-', lw=3, color='red', alpha=1, zorder=1, label=r'AMJJAS')
#plt.plot(df_seasonal.index, df_seasonal['ONDJFM'].rolling(10,center=True).mean(), ls='-', lw=3, color='blue', alpha=1, zorder=1, label=r'ONDJFM')
plt.plot(df_seasonal_fft.index, df_seasonal_fft['DJF'], ls='-', lw=3, color='blue', alpha=1, zorder=1, label=r'Winter')
plt.plot(df_seasonal_fft.index, df_seasonal_fft['MAM'], ls='-', lw=3, color='red', alpha=1, zorder=1, label=r'Spring')
plt.plot(df_seasonal_fft.index, df_seasonal_fft['JJA'], ls='-', lw=3, color='purple', alpha=1, zorder=1, label=r'Summer')
plt.plot(df_seasonal_fft.index, df_seasonal_fft['SON'], ls='-', lw=3, color='green', alpha=1, zorder=1, label=r'Autumn')
#plt.plot(df_seasonal_fft.index, df_seasonal_fft['AMJJAS'], ls='-', lw=3, color='red', alpha=1, zorder=1, label=r'AMJJAS')
#plt.plot(df_seasonal_fft.index, df_seasonal_fft['ONDJFM'], ls='-', lw=3, color='blue', alpha=1, zorder=1, label=r'ONDJFM')
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
# PLOT: seasonal differences
#------------------------------------------------------------------------------    
    
print('plotting seasonal differences ... ')   

figstr = station_code + '-' + 'wrapper' + '-' + 'seasonal-difference.png'
titlestr = 'GloSAT.p03: ' + station_name.upper() + ' (' + station_code + ') seasonal $T_g$ differences: decadal (FFT low pass)'
               
fig, ax = plt.subplots(figsize=(15,10))    
#plt.fill_between(df_yearly.index, -df_yearly_sd.Tg, +df_yearly_sd.Tg, ls='-', lw=1, color='black', alpha=0.2, zorder=0, label=r'Yearly $\mu\pm\sigma$')    
mask = np.isfinite(df_seasonal_fft)
mask = mask['DJF'] & mask['MAM'] & mask['JJA'] & mask['SON']
plt.plot(df_seasonal_fft.index[mask], df_seasonal_fft['DJF'][mask]-df_seasonal_fft['MAM'][mask], ls='-', lw=3, color='blue', alpha=1, zorder=1, label=r'DJF-MAM')
plt.plot(df_seasonal_fft.index[mask], df_seasonal_fft['DJF'][mask]-df_seasonal_fft['JJA'][mask], ls='-', lw=3, color='cyan', alpha=1, zorder=1, label=r'DJF-JJA')
plt.plot(df_seasonal_fft.index[mask], df_seasonal_fft['DJF'][mask]-df_seasonal_fft['SON'][mask], ls='-', lw=3, color='purple', alpha=1, zorder=1, label=r'DJF-SON')
plt.plot(df_seasonal_fft.index[mask], df_seasonal_fft['MAM'][mask]-df_seasonal_fft['JJA'][mask], ls='-', lw=3, color='red', alpha=1, zorder=1, label=r'MAM-JJA')
plt.plot(df_seasonal_fft.index[mask], df_seasonal_fft['MAM'][mask]-df_seasonal_fft['SON'][mask], ls='-', lw=3, color='orange', alpha=1, zorder=1, label=r'MAM-SON')
plt.plot(df_seasonal_fft.index[mask], df_seasonal_fft['JJA'][mask]-df_seasonal_fft['SON'][mask], ls='-', lw=3, color='yellow', alpha=1, zorder=1, label=r'JJA-SON')
plt.axhline(y=np.nanmean( df_seasonal_fft['DJF'][mask]-df_seasonal_fft['MAM'][mask] ), ls='--', lw=3, color='blue')
plt.axhline(y=np.nanmean( df_seasonal_fft['DJF'][mask]-df_seasonal_fft['JJA'][mask] ), ls='--', lw=3, color='cyan')
plt.axhline(y=np.nanmean( df_seasonal_fft['DJF'][mask]-df_seasonal_fft['SON'][mask] ), ls='--', lw=3, color='purple')
plt.axhline(y=np.nanmean( df_seasonal_fft['MAM'][mask]-df_seasonal_fft['JJA'][mask] ), ls='--', lw=3, color='red')
plt.axhline(y=np.nanmean( df_seasonal_fft['MAM'][mask]-df_seasonal_fft['SON'][mask] ), ls='--', lw=3, color='orange')
plt.axhline(y=np.nanmean( df_seasonal_fft['JJA'][mask]-df_seasonal_fft['SON'][mask] ), ls='--', lw=3, color='yellow')
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
