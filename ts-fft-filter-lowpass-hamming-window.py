#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: ts-fft-filter-lowpass-hamming-window.py
#------------------------------------------------------------------------------
# Version 0.2
# 26 July, 2021
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
# michael DOT a DOT taylor AT uea DOT ac DOT uk
#------------------------------------------------------------------------------

import numpy as np
import operator
from scipy import fftpack
from scipy.fftpack import fftfreq
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

#-----------------------------------------------------------------------------
# SETTINGS
#-----------------------------------------------------------------------------

fontsize = 16

#-----------------------------------------------------------------------------
# METHODS
#-----------------------------------------------------------------------------

def nextpowerof2(x):
    if x > 1:
        for i in range(1, int(x)):
            if (2 ** i >= x):
                return 2 ** i
    else:
        return 1

#-----------------------------------------------------------------------------
# SIGNAL
#-----------------------------------------------------------------------------

# GENERATE: signal from sum of sinusoids

f = np.array([ 440,880,1000,2000 ])                     # frequencies   
fc = 1500                                               # cutoff frequency 
M = 200                                                 # signal length 
Fs = 5000.0                                             # sampling rate
Ts = 1.0/Fs                                             # sampling interval 

n = np.arange(M)
y = np.zeros(M) 
for fk in f: 
    x = fk*Ts*np.arange(M)
    y += np.sin(2*np.pi*x) 
#   y += np.sin(2*np.pi*n*fk/Fs) 

# FFT zero-order estimate: low pass filter (no window)
 
y_fft = fftpack.fft(y)
y_fft_filtered = y_fft.copy()
freq = fftfreq(len(y), d=Ts)
# y_fft_filtered[ np.abs(freq) < fc ] = 0               # high-pass filter
y_fft_filtered[ np.abs(freq) > fc ] = 0                 # low-pass filter
y_filtered_no_window = fftpack.ifft(y_fft_filtered)     # filtered signal

# FILTER DESIGN: low pass filter (Hamming window)

L = M+1                                                 # filter length (M+1)
h_support = np.arange( -int((L-1)/2), int((L-1)/2)+1 )  # filter support
h_ideal = (2*fc/Fs)*np.sinc(2*fc*h_support/Fs)          # filter (ideal)
h = np.hamming(L).T*h_ideal                             # filter

# ZERO-PAD: (next power of 2 > L+M-1) signal and impulse-response

#Nfft = int(2**(np.ceil(np.log2(L+M-1))))
Nfft = nextpowerof2(L+M-1)
yzp = list(y) + list(np.zeros(Nfft-M+1))
hzp = list(h) + list(np.zeros(Nfft-L+1))    

# COMPUTE: FFT of signal and filter in freq domain

Y = fftpack.fft(yzp) # signal 
H = fftpack.fft(hzp) # filter

# COMPUTE: cyclic convolution (pairwise product) of signal and filter in freq domain

Z = np.multiply(Y, H)
y_filtered = fftpack.ifft(Z)[int(M/2):M+int((M)/2)]

#-----------------------------------------------------------------------------
# PLOTS
#-----------------------------------------------------------------------------

# PLOT: finite inpulse-reponse of ideal and designed filter

fig, ax = plt.subplots(figsize=(15,10))
plt.plot(h_ideal, ls='-', lw=5, color='red', alpha=0.5, label='h (ideal)')
plt.plot(h, ls='-', lw=3, color='lime', alpha=1, label='h (Hamming)')
plt.xlabel('Time', fontsize=fontsize)
plt.ylabel('Amplitude', fontsize=fontsize)
plt.title(str(fc) + ' Hz Low Pass Filter (Hann window): L=' + str(L), fontsize=fontsize)
plt.tick_params(labelsize=fontsize)    
plt.legend(loc='upper right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
fig.tight_layout()
plt.savefig('ts-fft-filter-hamming-design.png', dpi=300)
plt.close('all')

# PLOT: freq domain (signal and filter)

freq = fftfreq(len(Y), d=Ts)

fig, ax = plt.subplots(figsize=(15,10))
plt.subplot(211)
plt.plot(freq[0:int(len(freq)/2)], np.abs(H[0:int(len(freq)/2)])*np.max(np.abs(Y)), ls='--', lw=2, color='black', label='low pass filter (Hamming window)')
plt.stem(freq[0:int(len(freq)/2)], np.abs(Y[0:int(len(freq)/2)]), markerfmt=" ", basefmt="-b", label='FFT signal')
plt.fill_between(freq[0:int(len(freq)/2)], 0, np.abs(H[0:int(len(freq)/2)])*np.max(np.abs(Y)), facecolor='lime', alpha=0.2, label='FFT filter')
#plt.xlabel('Frequency, Hz', fontsize=fontsize)
plt.ylabel('Amplitude', fontsize=fontsize)
plt.title('Before filtering', fontsize=fontsize)
plt.tick_params(labelsize=fontsize)    
plt.legend(loc='upper right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
plt.subplot(212)
plt.stem(freq[0:int(len(freq)/2)], np.abs(Z[0:int(len(freq)/2)]), markerfmt=" ", basefmt="-b", label='FFT signal (low pass)')
plt.xlabel('Frequency, Hz', fontsize=fontsize)
plt.ylabel('Amplitude', fontsize=fontsize)
plt.title('After filtering', fontsize=fontsize)
plt.tick_params(labelsize=fontsize)    
plt.legend(loc='upper right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
fig.tight_layout()
plt.savefig('ts-fft-filter-hamming-spectrum.png', dpi=300)
plt.close('all')

# PLOT: time domain (signal and filtered signal)

fig, ax = plt.subplots(figsize=(15,10))
plt.plot(n, y, ls='-', lw=3, color='red', alpha=0.5, label='signal')
#plt.plot(n, y_filtered_no_window, ls='--', lw=1, color='black', alpha=1, label='signal: FFT low pass (no window) cut-off=' + str(fc) + ' Hz')
plt.plot(n, y_filtered, ls='-', lw=3, color='lime', alpha=1, label='signal: FFT low pass (Hamming window) cut-off=' + str(fc) + ' Hz')
plt.plot(n, y - y_filtered, ls='-', lw=3, color='teal', alpha=1, label='signal: FFT high pass (Hamming window) cut-off=' + str(fc) + ' Hz')
plt.xlabel('Time', fontsize=fontsize)
plt.ylabel('Amplitude', fontsize=fontsize)
plt.title('SoS: [440, 880, 1000, 2000] Hz', fontsize=fontsize)
plt.tick_params(labelsize=fontsize)    
plt.legend(loc='upper right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
fig.tight_layout()
plt.savefig('ts-fft-filter-hamming-signal.png', dpi=300)
plt.close('all')

# PLOT: time domain (signal and filtered signal)

fig, ax = plt.subplots(figsize=(15,10))
plt.plot(n, y_filtered, ls='-', lw=3, color='lime', alpha=1, label='signal: FFT low pass (Hamming window) cut-off=' + str(fc) + ' Hz')
plt.plot(n, y_filtered_no_window, ls='--', lw=1, color='black', alpha=1, label='signal: FFT low pass (no window) cut-off=' + str(fc) + ' Hz')
plt.xlabel('Time', fontsize=fontsize)
plt.ylabel('Amplitude', fontsize=fontsize)
plt.title('SoS: [440, 880, 1000, 2000] Hz', fontsize=fontsize)
plt.tick_params(labelsize=fontsize)    
plt.legend(loc='upper right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
fig.tight_layout()
plt.savefig('ts-fft-filter-hamming-signal-versus-no-window.png', dpi=300)
plt.close('all')

#-----------------------------------------------------------------------------
print('** END')


























    
    
