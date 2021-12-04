#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: seasonal-means-and-nao.py
#------------------------------------------------------------------------------
# Version 0.1
# 30 November, 2021
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
#import matplotlib.cm as cm
#from matplotlib import colors as mcol
#from matplotlib.cm import ScalarMappable
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
import filter_cru_dft as cru # CRU DFT filter
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# SETTINGS: 
#------------------------------------------------------------------------------

fontsize = 16
use_dark_theme = False

#baseline_start = 1981
#baseline_end = 2010
baseline_start = 1961
baseline_end = 1990
use_fahrenheit = False
if use_fahrenheit: 
    temperature_unit = 'F'
else:
    temperature_unit = 'C'

use_anomalies = True

nsmooth = 60                 # 5yr MA monthly
nfft = 16                    # power of 2 for the DFT
loess_frac = 0.2             # LOESS window 

station_code = '024581'     # Uppsala 

#----------------------------------------------------------------------------
# DARK THEME
#----------------------------------------------------------------------------

if use_dark_theme == True:
    
    matplotlib.rcParams['text.usetex'] = False
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

    print('using Seasborn graphics ...')
    
#    matplotlib.rcParams['text.usetex'] = False
#    plt.rc('text',color='black')
#    plt.rc('lines',color='black')
#    plt.rc('patch',edgecolor='black')
#    plt.rc('grid',color='lightgray')
#    plt.rc('xtick',color='black')
#    plt.rc('ytick',color='black')
#    plt.rc('axes',labelcolor='black')
#    plt.rc('axes',facecolor='white')    
#    plt.rc('axes',edgecolor='black')
#    plt.rc('figure',facecolor='white')
#    plt.rc('figure',edgecolor='white')
#    plt.rc('savefig',edgecolor='white')
#    plt.rc('savefig',facecolor='white')

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
    
#    y = x - np.nanmean(x)
#    w = scipy.fftpack.rfft(y) 
#    spectrum = w**2 
#    cutoff_idx = spectrum < (spectrum.max() * (1 - np.exp(-span / 2000)))
#    w[cutoff_idx] = 0    
#    y_filtered = fftpack.irfft(w)
#    x_filtered = y_filtered + np.nanmean(x)

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
    
def linear_regression_ols(x,y):
    
    regr = linear_model.LinearRegression()
    # regr = TheilSenRegressor(random_state=42)
    # regr = RANSACRegressor(random_state=42)
    
    X = x.reshape(len(x),1)
    xpred = np.linspace(X.min(),X.max(),len(X)) # dummy var spanning [xmin,xmax]        
    regr.fit(X, y)
    ypred = regr.predict(xpred.reshape(-1, 1))
    slope = regr.coef_[0]
    intercept = regr.intercept_     
        
    return xpred, ypred, slope, intercept

#------------------------------------------------------------------------------    
# LOAD: GloSAT temperature archive: CRUTEM5.0.1.0
#------------------------------------------------------------------------------
            
print('loading temperatures ...')
        
df_temp = pd.read_pickle('DATA/df_temp.pkl', compression='bz2')    
df = df_temp[df_temp['stationcode']==station_code]
station_name = df['stationname'].unique()[0]
    
# SET: time axis container
        
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
    
loess = sm.nonparametric.lowess(df['MA'].values[mask], df['MA'].index[mask], frac=loess_frac)       
da = pd.DataFrame({'LOESS':loess[:,1]}, index=df['MA'].index[mask])
df['LOESS'] = da                     

# CALCULATE: FFT low pass

da = pd.DataFrame({'FFT':smooth_fft(df['MA'].values[mask], nfft)}, index=df['MA'].index[mask])
df['FFT'] = da.rolling(nsmooth, center=True).mean()
# df['FFT'] = da[(da.index>da.index[int(nfft/2)]) & (da.index<=da.index[-int(nfft/2)])] # edge effect truncation
    
#------------------------------------------------------------------------------    
# LOAD: NAO indices
#------------------------------------------------------------------------------
            
print('loading NAO indices ...')

#Year	Mon	NAO
#1658	Dec	-0.30
#1659	Jan	0.16

nao_file = 'DATA/naomonjurg.dat'
nheader = 26
f = open(nao_file)
lines = f.readlines()
years = []
months = [] 
obs = []

month_dict = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}

for i in range(nheader,len(lines)):
    words = lines[i].split()   
    year = int(words[0])
    month = month_dict[ words[1] ]
    val = float(words[2])
    years.append(year)                                     
    months.append(month)            
    obs.append(val)            
f.close()    
obs = np.array(obs)

da = pd.DataFrame()             
da['year'] = years
da['month'] = months
da['nao'] = obs

# TRIM: to start of Pandas datetime range ( if needed )

da = da[da.year >= 1678].reset_index(drop=True)
dates = [ pd.to_datetime( str(da.year[i]) + '-' + str(da.month[i]).zfill(2) + '-01', format='%Y-%m-%d' ) for i in range(len(da)) ]
df_nao = pd.DataFrame({'nao':da.nao.values}, index=dates )

# MERGE: anomaly and NAO timeseries

df_monthly = df.merge(df_nao, left_index=True, right_index=True)
t_monthly = df_monthly.index
ts_monthly = df_monthly.Tg.values
nao_monthly = df_monthly.nao.values

plt.plot(df_monthly.index, df_monthly.Tg)
plt.plot(df_monthly.index, df_monthly.nao)
plt.plot(df_monthly.index, df_monthly.MA)
plt.plot(df_monthly.index, df_monthly.FFT)
plt.plot(df_monthly.index, df_monthly.LOESS, color='black')

# RESAMPLE: seasonal weighted means
    
#    month_length = df_uppsala_xr.index.dt.days_in_month
#    weights = month_length.groupby('index.season') / month_length.groupby('index.season').sum()
#    df_seasonal_xr = (df_uppsala_xr * weights).groupby('index.season').sum(dim='index')    
    
# EXTRACT: seasonal components ( D from first year only --> N-1 seasonal estimates )

trim_months = len(t_monthly)%12
da = pd.DataFrame({'Tg':ts_monthly[:-1-trim_months]}, index=t_monthly[:-1-trim_months])         
db = pd.DataFrame({'Tg':nao_monthly[:-1-trim_months]}, index=t_monthly[:-1-trim_months])         

da_DJF = ( da[da.index.month==12]['Tg'].values + da[da.index.month==1]['Tg'].values[1:] + da[da.index.month==2]['Tg'].values[1:] ) / 3
da_MAM = ( da[da.index.month==3]['Tg'].values[1:] + da[da.index.month==4]['Tg'].values[1:] + da[da.index.month==5]['Tg'].values[1:] ) / 3
da_JJA = ( da[da.index.month==6]['Tg'].values[1:] + da[da.index.month==7]['Tg'].values[1:] + da[da.index.month==8]['Tg'].values[1:] ) / 3
da_SON = ( da[da.index.month==9]['Tg'].values[1:] + da[da.index.month==10]['Tg'].values[1:] + da[da.index.month==11]['Tg'].values[1:] ) / 3
da_ONDJFM = ( da[da.index.month==10]['Tg'].values[1:] + da[da.index.month==11]['Tg'].values[1:] + da[da.index.month==12]['Tg'].values + da[da.index.month==1]['Tg'].values[1:] + da[da.index.month==2]['Tg'].values[1:] + da[da.index.month==3]['Tg'].values[1:] ) / 6
da_AMJJAS = ( da[da.index.month==4]['Tg'].values[1:] + da[da.index.month==5]['Tg'].values[1:] + da[da.index.month==6]['Tg'].values[1:] + da[da.index.month==7]['Tg'].values[1:] + da[da.index.month==8]['Tg'].values[1:] + da[da.index.month==9]['Tg'].values[1:] ) / 6

db_DJF = ( db[db.index.month==12]['Tg'].values + db[db.index.month==1]['Tg'].values[1:] + db[db.index.month==2]['Tg'].values[1:] ) / 3
db_MAM = ( db[db.index.month==3]['Tg'].values[1:] + db[db.index.month==4]['Tg'].values[1:] + db[db.index.month==5]['Tg'].values[1:] ) / 3
db_JJA = ( db[db.index.month==6]['Tg'].values[1:] + db[db.index.month==7]['Tg'].values[1:] + db[db.index.month==8]['Tg'].values[1:] ) / 3
db_SON = ( db[db.index.month==9]['Tg'].values[1:] + db[db.index.month==10]['Tg'].values[1:] + db[db.index.month==11]['Tg'].values[1:] ) / 3
db_ONDJFM = ( db[db.index.month==10]['Tg'].values[1:] + db[db.index.month==11]['Tg'].values[1:] + db[db.index.month==12]['Tg'].values + db[db.index.month==1]['Tg'].values[1:] + db[db.index.month==2]['Tg'].values[1:] + db[db.index.month==3]['Tg'].values[1:] ) / 6
db_AMJJAS = ( db[db.index.month==4]['Tg'].values[1:] + db[db.index.month==5]['Tg'].values[1:] + db[db.index.month==6]['Tg'].values[1:] + db[db.index.month==7]['Tg'].values[1:] + db[db.index.month==8]['Tg'].values[1:] + db[db.index.month==9]['Tg'].values[1:] ) / 6

t_seasonal = pd.date_range( start=str(da.index[0].year+1), end=str(da.index[-1].year), freq='AS')
df_seasonal = pd.DataFrame({
    'da_DJF':da_DJF, 'da_MAM':da_MAM, 'da_JJA':da_JJA, 'da_SON':da_SON, 'da_ONDJFM':da_ONDJFM, 'da_AMJJAS':da_AMJJAS, 
    'db_DJF':db_DJF, 'db_MAM':db_MAM, 'db_JJA':db_JJA, 'db_SON':db_SON, 'db_ONDJFM':db_ONDJFM, 'db_AMJJAS':db_AMJJAS}, 
    index = t_seasonal)     

# SELECT: season

season = 'DJF'
#season = 'MAM'
#season = 'JJA'
#season = 'SON'
#season = 'ONDJFM'
#season = 'AMJJAS'

if season == 'DJF':
    da_season = 'da_DJF'
    db_season = 'db_DJF'
elif season == 'MAM':
    da_season = 'da_MAM'
    db_season = 'db_MAM'
elif season == 'JJA':
    da_season = 'da_JJA'
    db_season = 'db_JJA'
elif season == 'SON':
    da_season = 'da_SON'
    db_season = 'db_SON'
elif season == 'ONDJFM':
    da_season = 'da_ONDJFM'
    db_season = 'db_ONDJFM'
elif season == 'AMJJAS':
    da_season = 'da_AMJJAS'
    db_season = 'db_AMJJAS'

mask = np.isfinite(df_seasonal[da_season]) & np.isfinite(df_seasonal[db_season])
tem = df_seasonal[da_season][mask]
nao = df_seasonal[db_season][mask]
t_tem = tem.index
t_nao = nao.index
t_tem1 = tem[tem.index.year<=1740].index
t_nao1 = nao[nao.index.year<=1740].index
t_tem2 = tem[tem.index.year>1740].index
t_nao2 = nao[nao.index.year>1740].index
t_tem3 = tem[ (tem.index.year>=1761) & (tem.index.year<=1890) ].index
t_nao3 = nao[ (nao.index.year>=1761) & (nao.index.year<=1890) ].index
t_tem4 = tem[ (tem.index.year>=1891) & (tem.index.year<=2020) ].index
t_nao4 = nao[ (nao.index.year>=1891) & (nao.index.year<=2020) ].index

#------------------------------------------------------------------------------
# OLS: linear regression
#------------------------------------------------------------------------------

X1 = tem[tem.index.year<=1740].values
Y1 = nao[nao.index.year<=1740].values
X2 = tem[tem.index.year>1740].values
Y2 = nao[nao.index.year>1740].values
X3 = tem[ (tem.index.year>=1761) & (tem.index.year<=1890) ].values
Y3 = nao[ (nao.index.year>=1761) & (nao.index.year<=1890) ].values
X4 = tem[ (tem.index.year>=1891) & (tem.index.year<=2020) ].values
Y4 = nao[ (nao.index.year>=1891) & (nao.index.year<=2020) ].values

minval = np.nanmin([np.nanmin(X1),np.nanmin(Y1),np.nanmin(X2),np.nanmin(Y2)])
maxval = np.nanmax([np.nanmax(X1),np.nanmax(Y1),np.nanmax(X2),np.nanmax(Y2)])        
corrcoef_old = scipy.stats.pearsonr(X1, Y1)[0]
corrcoef_new = scipy.stats.pearsonr(X2, Y2)[0]
OLS_X_old, OLS_Y_old, OLS_slope_old, OLS_intercept_old = linear_regression_ols(X1, Y1)
OLS_X_new, OLS_Y_new, OLS_slope_new, OLS_intercept_new = linear_regression_ols(X2, Y2)

#==============================================================================
# PLOTS
#==============================================================================

if use_dark_theme == True:
    default_color = 'white'
else:    
    default_color = 'black'    	
    
#------------------------------------------------------------------------------    
# PLOT: seasonal timeseries
#------------------------------------------------------------------------------    
    
print('plotting seasonal timeseries ... ')   
                            
figstr = station_code + '-' + 'anomaly' + '-' + 'vs' + '-' + 'nao' + '-' 'timeseries' + '-' + season + '.png'
titlestr = 'GloSAT.p03: ' + station_name.upper() + ' (' + station_code + ') ' + season + ' seasonal mean $T_g$ anomaly (from 1961-1990) vs NAO'
               
fig, ax = plt.subplots(figsize=(15,10))    
plt.fill_between( t_tem, tem, ls='-', lw=0.5, color='blue', alpha=0.8, zorder=1, label=r'$T_{g}$')
plt.fill_between( t_nao, nao, ls='-', lw=0.5, color='red', alpha=0.8, zorder=1, label=r'$NAO$')
plt.axvline( x=pd.to_datetime('1740-01-01', format='%Y-%m-%d'), ls='dashed', lw=2, color='black', label='1739/1740')
plt.axhline( y=0, ls='dashed', lw=1, color='black')
plt.xlim( pd.to_datetime(t_tem[0], format='%Y-%m-%d'), pd.to_datetime('1760-01-01', format='%Y-%m-%d') )
plt.xlabel('Year', fontsize=fontsize)
plt.ylabel(season + ' seasonal mean', fontsize=fontsize)
plt.title(titlestr, fontsize=fontsize)
plt.tick_params(labelsize=fontsize)    
plt.legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
fig.tight_layout()
plt.savefig(figstr, dpi=300, bbox_inches='tight')
plt.close('all')

print('plotting seasonal timeseries ( 130 intervals ) ... ')   
                            
figstr = station_code + '-' + 'anomaly' + '-' + 'vs' + '-' + 'nao' + '-' 'timeseries' + '-' + season + '-' + '130 years' + '.png'
titlestr = 'GloSAT.p03: ' + station_name.upper() + ' (' + station_code + ') ' + season + ' seasonal mean $T_g$ anomaly (from 1961-1990) vs NAO'
               
fig, ax = plt.subplots(2,1, figsize=(15,10))    
ax[0].fill_between( t_tem3, X3, ls='-', lw=0.5, color='blue', alpha=0.8, zorder=1, label=r'$T_{g}$')
ax[0].fill_between( t_nao3, Y3, ls='-', lw=0.5, color='red', alpha=0.8, zorder=1, label=r'$NAO$')
ax[0].axhline( y=0, ls='dashed', lw=1, color='black')
ax[0].set_xlabel('Year', fontsize=fontsize)
ax[0].set_ylabel(season + ' seasonal mean', fontsize=fontsize)
ax[0].set_title(titlestr, fontsize=fontsize)
ax[0].tick_params(labelsize=fontsize)    
ax[1].fill_between( t_tem4, X4, ls='-', lw=0.5, color='blue', alpha=0.8, zorder=1, label=r'$T_{g}$')
ax[1].fill_between( t_nao4, Y4, ls='-', lw=0.5, color='red', alpha=0.8, zorder=1, label=r'$NAO$')
ax[1].axhline( y=0, ls='dashed', lw=1, color='black')
ax[1].set_xlabel('Year', fontsize=fontsize)
ax[1].set_ylabel(season + ' seasonal mean', fontsize=fontsize)
ax[1].tick_params(labelsize=fontsize)    
ax[1].legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
fig.tight_layout()
plt.savefig(figstr, dpi=300, bbox_inches='tight')
plt.close('all')

#------------------------------------------------------------------------------    
# PLOT: seasonal correlation
#------------------------------------------------------------------------------    

print('plotting seasonal correlation ... ')   
   
figstr = station_code + '-' + 'anomaly' + '-' + 'vs' + '-' + 'nao' + '-' 'correlation' + '-' + season + '.png'
titlestr = 'GloSAT.p03: ' + station_name.upper() + ' (' + station_code + ') ' + season + ' seasonal mean $T_g$ anomaly (from 1961-1990) vs NAO'
               
fig, ax = plt.subplots(figsize=(15,10))    
plt.plot(OLS_X_new, OLS_Y_new, color='blue', ls='-', lw=2, zorder=0, label=r'OLS ($\alpha x + \beta$)' + r': $\alpha$=' + str(np.round(OLS_slope_new,3)) + ' ' + r'$\beta$=' + str(np.round(OLS_intercept_new,3)) + ' ' + r'(Pearson $\rho$='+str(np.round(corrcoef_new,3)) + ')' )
plt.plot(OLS_X_old, OLS_Y_old, color='navy', ls='-', lw=2, zorder=0, label=r'OLS ($\alpha x + \beta$)' + r': $\alpha$=' + str(np.round(OLS_slope_old,3)) + ' ' + r'$\beta$=' + str(np.round(OLS_intercept_old,3)) + ' ' + r'(Pearson $\rho$='+str(np.round(corrcoef_old,3)) + ')')
plt.scatter(X2, Y2, alpha=0.5, marker='o', color='blue', s=50, facecolor='lightblue', ls='-', lw=1, zorder=1, label=r'> 1740')
plt.scatter(X1, Y1, alpha=0.5, marker='o', color='navy', s=50, facecolor='navy', ls='-', lw=1, zorder=1, label=r'$\leq$ 1740')
plt.scatter(tem[tem.index.year==1740].values, nao[nao.index.year==1740].values, alpha=0.5, marker='o', color='red', s=100, facecolor='red', ls='-', lw=1, zorder=1, label=r'1739/1740')
ax.plot([minval,maxval], [minval,maxval], color='black', ls='--', lw=1, alpha=1, zorder=0)    
ax.set_xlim(minval, maxval)
ax.set_ylim(minval, maxval)
ax.set_aspect('equal') 
ax.xaxis.grid(True, which='minor')      
ax.yaxis.grid(True, which='minor')  
ax.xaxis.grid(True, which='major')      
ax.yaxis.grid(True, which='major')  
ax.tick_params(axis='both', which='major', labelsize=fontsize)
plt.xlabel(r'$T_{g}$ ' + season + ' seasonal mean', fontsize=fontsize)
plt.ylabel(r'$NAO$ ' + season + ' seasonal mean', fontsize=fontsize)
plt.title(titlestr, fontsize=fontsize)
plt.tick_params(labelsize=fontsize)    
plt.legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
fig.tight_layout()
plt.savefig(figstr, dpi=300, bbox_inches='tight')
plt.close('all')
              
#------------------------------------------------------------------------------
print('** END')
