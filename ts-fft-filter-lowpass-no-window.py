#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: ts-fft-filter-lowpass-no-window.py
#------------------------------------------------------------------------------
# Version 0.1
# 25 July, 2021
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
# michael DOT a DOT taylor AT uea DOT ac DOT uk
#------------------------------------------------------------------------------

import numpy as np
from scipy import fftpack
from scipy.fftpack import fftfreq
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

#-----------------------------------------------------------------------------
# SETTINGS
#-----------------------------------------------------------------------------

fontsize = 16

#-----------------------------------------------------------------------------
# GENERATE: signal from sum of sinusoids
#-----------------------------------------------------------------------------

f = np.array([ 1,4,6,9 ])                 # frequencies (kHz)
a = np.array([ 3,1,0.5,0.5 ])               # coefficients
fc = 7                                  # cut-off frequency
M = 4000                                # signal length 
Fs = 2000.0                             # sampling rate
Ts = 1.0/Fs                             # Sampling interval 

#x = np.arange(0,1,Ts)
#y = np.zeros(Fs) 

n = np.arange(M)
y = np.zeros(M) 
for fk, ak in zip(f,a): 
    x = fk*Ts*np.arange(M)
    y += ak*np.sin(2*np.pi*x) 
#   y += ak*np.sin(2*np.pi*fk*x) 

# FFT zero-order estimate: low pass filter (no window)

yf = fftpack.fft(y)
yf_filtered = yf.copy()
freq = fftfreq(len(y), d=Ts)
#yf_filtered[ np.abs(freq) < fc ] = 0 # high-pass filter
yf_filtered[ np.abs(freq) > fc ] = 0 # low-pass filter
y_filtered = fftpack.ifft(yf_filtered) # filtered signal in time domain

#-----------------------------------------------------------------------------
# PLOTS
#-----------------------------------------------------------------------------

# PLOT: freq domain (signal and filter)

ymax = np.max( np.abs(yf)/np.sum(np.abs(yf)) )

fig, ax = plt.subplots(figsize=(15,10))
plt.subplot(211)
plt.plot([0,fc], [ymax,ymax], ls='--', lw=2, color='black')
plt.plot([fc,10], [0,0], ls='--', lw=2, color='black')
plt.plot([fc,fc], [0,ymax], ls='--', lw=2, color='black', label='low pass filter (cut-off=' + str(fc) + ' kHz)')
plt.fill_between(np.linspace(0,fc), 0, ymax, color='lime', alpha=0.2, label='FFT filter')
plt.stem(freq, np.abs(yf)/np.sum(np.abs(yf)), 'b', markerfmt=" ", basefmt="b", linefmt="-", label='FFT signal')
plt.xlim(0, 10)
plt.title('Before filtering', fontsize=fontsize)
plt.tick_params(labelsize=fontsize)    
# plt.xlabel('Frequency, kHz', fontsize=fontsize)
plt.ylabel('FFT amplitude (relative)', fontsize=fontsize)
plt.legend(loc='upper right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)   
plt.subplot(212)
plt.stem(freq, np.abs(yf_filtered)/np.sum(np.abs(yf)), 'b', markerfmt=" ", basefmt="b", linefmt="-", label='FFT signal (low pass)')
plt.xlim(0, 10)
plt.tick_params(labelsize=fontsize)    
plt.title('After filtering', fontsize=fontsize)
plt.xlabel('Frequency, kHz', fontsize=fontsize)
plt.ylabel('FFT amplitude (relative)', fontsize=fontsize)
plt.legend(loc='upper right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)   
fig.tight_layout()
plt.savefig('ts-fft-filter-no-window-spectrum.png', dpi=300)
plt.close('all')

# PLOT: time domain (signal and filtered signal)

fig, ax = plt.subplots(figsize=(15,10))
plt.plot(n, y, ls='-', lw=3, color='red', label='signal')
plt.plot(n, y_filtered, ls='-', lw=3, color='lime', label='FFT low pass (cut-off=' + str(fc) + ' kHz)')
plt.plot(n, y - y_filtered, ls='-', lw=3, color='teal', label='FFT high pass (cut-off=' + str(fc) + ' kHz)')
plt.tick_params(labelsize=fontsize)    
plt.xlabel('Time', fontsize=fontsize)
plt.ylabel('Magnitude', fontsize=fontsize)
plt.title('SoS: [1,4,6,9] kHz: FFT filtering (no window)', fontsize=fontsize)
plt.legend(loc='upper right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)   
fig.tight_layout()
plt.savefig('ts-fft-filter-no-window-signal.png', dpi=300)
plt.close('all')

#-----------------------------------------------------------------------------
print('** END')








