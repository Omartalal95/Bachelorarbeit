# -*- coding: utf-8 -*-
"""
Spyder Editor

Dies ist eine temporäre Skriptdatei.
"""
import numpy as np
import librosa as lb
from heapq import nlargest

from scipy.fftpack import fft
from scipy.fftpack import dct
from scipy.fftpack import idct
from scipy import signal


import matplotlib.pyplot as plt


############ Signals ###############################
x, sr = lb.core.load('H:\\01_einfaches modell\\1.01_london_phill_dataset_multi\\cello\\cello_A2_05_pianissimo_arco-normal.mp3')
y = np.arange(0,len(x))*x

########## Constants ###############################
T = 1/sr
alpha = 0.4
gamma = 0.99

######## FFT ####################
X = np.fft.fft(x, n=8192)
Y = np.fft.fft(y, n=8192)

####### Product Spectrum #########
power_product = X.real*Y.real + X.imag*Y.imag

####### Group Delay Spectrum ################
group_delay = power_product/(np.abs(X)**2) 

###### Modified Group Delay Spectrum #########
Log_Amp_filterd = dct(signal.medfilt(np.log10(np.abs(X)), 7))[0:31]
sm_spectra = idct(Log_Amp_filterd, n = len(X))
temp = power_product/np.power(np.abs(sm_spectra), 2*gamma)
temp_abs = np.abs(temp)
modg = (temp/temp_abs)*np.power(temp_abs, alpha) 

######### Frequecy bands ##########################
freq = np.fft.fftfreq(len(X), T)

#############################################################
Note = 'Cello, A2, Pianissimo Arco Normal'

############ FFT Plot ####################################### 
fig, ax = plt.subplots()
plt.plot( freq, np.abs(X)/np.max(np.abs(X)), linewidth=2.0)
plt.scatter([freq[40]], [(np.abs(X)/np.max(np.abs(X)))[40]], marker='o', color='k', alpha=1, linewidths=4)
plt.scatter([freq[81], freq[122], freq[163]], [ 
                                                        (np.abs(X)/np.max(np.abs(X)))[81],
                                                        (np.abs(X)/np.max(np.abs(X)))[122], 
                                                        (np.abs(X)/np.max(np.abs(X)))[163]], 
                                                        marker='o', color='red', alpha=1, linewidths=4)
plt.xlabel('f in Hz', fontsize=13)
plt.ylabel('Normalized Amplitude')
plt.title('FFT Spectrum \n '+ Note ,fontweight="bold")
plt.xlim(0, 2000)
plt.ylim(0, 1)
plt.grid(True)
#plt.text(0.3, 0.97, 'Cello, A2, Pianissimo Arco Normal', horizontalalignment='left', verticalalignment='top', 
#         transform=ax.transAxes, bbox=dict(facecolor='red', alpha=0.2), fontsize='large', fontstyle='oblique')
#plt.annotate('Fundamental Frequency: 110 Hz', xy=(freq[41], (np.abs(X)/np.max(np.abs(X)))[41])
#             , xytext=(600, 0.6),
#            arrowprops = dict(facecolor="black", width=0.85, headwidth=7, shrink=0), bbox=dict(facecolor='red', alpha=0.2)
#            , fontsize='large', fontstyle='oblique')
plt.legend(['FFT Spectrum', 'Fundamental Frequency: 110 Hz', 'Harmonics'], loc=2)
############ Plot Group Delay########################
fig, ax = plt.subplots()
plt.plot( freq, group_delay/np.max(np.abs(group_delay)), linewidth=2.0)
plt.scatter([freq[40]], [(group_delay/np.max(np.abs(group_delay)))[40]], marker='o', color='k', alpha=1, linewidths=4)
plt.scatter([freq[81], freq[122], freq[163]], [ 
                                                        (group_delay/np.max(np.abs(group_delay)))[81],
                                                        (group_delay/np.max(np.abs(group_delay)))[122], 
                                                        (group_delay/np.max(np.abs(group_delay)))[163]], 
                                                        marker='o', color='red', alpha=1, linewidths=4)
                                                                                 
plt.xlabel('f in Hz', fontsize=13)
plt.ylabel('Normalized Amplitude', fontsize=11)
plt.title('Group Delay Spectrum \n ' + Note ,fontweight="bold")
plt.xlim(0, 10000)
plt.ylim(-0.75, 1)
plt.grid(True)
#plt.text(0.2, 0.2, 'Cello, A2, Pianissimo Arco Normal', horizontalalignment='left', verticalalignment='top', 
#         transform=ax.transAxes, bbox=dict(facecolor='red', alpha=0.2), fontsize='large', fontstyle='oblique')
#plt.annotate('Fundamental Frequency: 110 Hz', xy=(freq[41], (group_delay/np.max(np.abs(group_delay)))[41])
#             , xytext=(1100, 0.2), arrowprops = dict(facecolor="black", width=0.85, headwidth=7, shrink=0), bbox=dict(facecolor='red', alpha=0.2)
#            , fontsize='large', fontstyle='oblique')

plt.legend(['MODG Spectrum', 'Fundamental Frequency: 110 Hz', 'Harmonics'], loc=2)
################ Plot MODG#######################################################
fig, ax = plt.subplots()
plt.plot( freq, modg/np.max(np.abs(modg)), linewidth=2.0)
plt.scatter([freq[40]], [(modg/np.max(np.abs(modg)))[40]], marker='o', color='k', alpha=1, linewidths=4)
plt.scatter([freq[81], freq[122], freq[163]], [(modg/np.max(np.abs(modg)))[81],
                                              (modg/np.max(np.abs(modg)))[122], 
                                              (modg/np.max(np.abs(modg)))[163]], marker='o', color='red', alpha=1,
                                                                                 linewidths=4)
plt.xlabel('f in Hz', fontsize=13)
plt.ylabel('Normalized Amplitude', fontsize=11)
plt.title('Modified Group Delay Spectrum \n '+ Note ,fontweight="bold")
plt.xlim(0, 2000)
plt.ylim(0, 1)
plt.grid(True)
#plt.text(0.3, 0.5, 'Cello, A2, Pianissimo Arco Normal', horizontalalignment='left', verticalalignment='top', 
#         transform=ax.transAxes, bbox=dict(facecolor='red', alpha=0.2), fontsize='large', fontstyle='oblique')
#plt.annotate('Fundamental Frequency: 110 Hz', xy=(freq[41], (modg/np.max(np.abs(modg)))[41])
#             , xytext=(700, 0.6), arrowprops = dict(facecolor="black", width=0.85, headwidth=7, shrink=0), bbox=dict(facecolor='red', alpha=0.2)
#             , fontsize='large', fontstyle='oblique')
plt.legend(['MODG Spectrum', 'Fundamental Frequency: 110 Hz', 'Harmonics'], loc=2)

################# Plot Product Spectrum################################
fig, ax = plt.subplots()
plt.plot( freq, power_product/np.max(np.abs(power_product)), linewidth=2.0)
plt.scatter([freq[40]], [(power_product/np.max(np.abs(power_product)))[40]], marker='o', color='k', alpha=1, linewidths=4)
plt.scatter([freq[81], freq[122], freq[163]], [(power_product/np.max(np.abs(power_product)))[81],
                                              (power_product/np.max(np.abs(power_product)))[122], 
                                              (power_product/np.max(np.abs(power_product)))[163]], marker='o', color='red', alpha=1,
                                                                                 linewidths=4)
plt.xlabel('f in Hz', fontsize=13)
plt.ylabel('Normalized Amplitude', fontsize=11)
plt.title('Product Spectrum \n ' + Note ,fontweight="bold")
plt.xlim(0, 2000)
plt.ylim(0, 1)
plt.grid(True)
#plt.text(0.3, 0.5, 'Cello, A2, Pianissimo Arco Normal', horizontalalignment='left', verticalalignment='top', 
#         transform=ax.transAxes, bbox=dict(facecolor='red', alpha=0.2), fontsize='large', fontstyle='oblique')
#plt.annotate('Fundamental Frequency: 110 Hz', xy=(freq[41], (modg/np.max(np.abs(modg)))[41])
#             , xytext=(700, 0.6), arrowprops = dict(facecolor="black", width=0.85, headwidth=7, shrink=0), bbox=dict(facecolor='red', alpha=0.2)
#             , fontsize='large', fontstyle='oblique')
plt.legend(['Product Spectrum', 'Fundamental Frequency: 110 Hz', 'Harmonics'], loc=2)



#freq[np.argmax(np.abs(X))]
#freq[np.argmax(power_product)]
#freq[np.argmax(group_delay)]/freq[np.argmax(power_product)]
#
#nlargest(6, np.abs(X)/np.max(np.abs(X)))
#np.where(np.abs(X)/np.max(np.abs(X)) == 0.6805246)
#freq[41]
#(np.abs(X)/np.max(np.abs(X)))[41]