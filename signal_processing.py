# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 09:06:30 2019

@author: Omar Elaiashy

Dieses Skript enthält die Signaleverarbeitungsmethode die in diesem Bacheloerarbeit 
verwendet wird.

1 - STFT-Spectrogram (Librosa)
2 - Mel-Spectrogram (Librosa) 
3 - Modified group delay function Spectrum 
   (Paper: Product of Power Spectrum and Group delay function for Speech recognition)
4 - Mel-MODGF Spectrogram    
4 - Product of Power Spectrogram
   (Paper: Product of Power Spectrum and Group delay function for Speech recognition)
5 - Recurrene Plot of Recurrence Plots      
"""
import os
import numpy as np
from scipy.fftpack import fft
from scipy.fftpack import dct
from scipy.fftpack import idct
from scipy import signal
import librosa as lb
import librosa.display
import matplotlib.pyplot as plt
from config import*

##Pre emphasising ,Framing and windowing
def make_windowed_frames(sig):
    ##Global Parameters
    #global pre_emphasis, N_frame, hop_length
    
    ##Pre emphasising the Signal
    emp_sig = np.append(sig[0], sig[1:] - pre_emphasis*sig[:-1])
    
    ##Framing the Signal
    len_sig = len(sig)
    N_frames = int(len_sig/hop_length + 1)   #Number of Frames 
    frames = np.ndarray((N_frames, N_frame), dtype= float)
    
    ##Windowing the Signal to decreace leakage effect
    w = signal.hamming(N_frame).reshape((1,-1))
    
    for i in np.arange(N_frames):
        for j in np.arange(N_frame):
            if (i*hop_length + j) < len_sig:
                frames[i][j] = emp_sig[i*hop_length + j]*w[0, j]
            else:
                frames[i][j] = 0 
    return frames

def stft_spectrogram(sig,norm_db):
    #global N_fft, hop_length, N_frame
    #sig /= sig.max() 
    y_stft = np.abs(lb.stft(sig, n_fft = N_fft,
                   hop_length = hop_length, win_length = N_frame))
    if norm_db:
        #return lb.util.normalize(y_stft, norm = np.inf)
        return lb.amplitude_to_db((y_stft + offset), np.max)
    else:
        return y_stft
    
def mel_spectrogram(sig,norm_db):
    #global sr, N_fft, hop_length, N_frame, n_mels, f_max
    #sig /= sig.max()
    y_mel = lb.feature.melspectrogram(y=sig, sr=sr, n_fft=N_fft,hop_length=hop_length,
                                      fmax = f_max, n_mels = n_mels)
    if norm_db:
        #return lb.util.normalize(y_mel, norm = np.inf)
        return lb.power_to_db((y_mel + offset), np.max)
    else:
        return y_mel
    
def modg_spectrogram(sig, norm_db):
    """
    It calulates the Modified Group delay Spectrogram (normalized in dB)
    and Power Spectrogram (normalized in dB)
    
    Input:
        sig: Input Signal
        
        norm_db: To normalize MODG in dB
    Output:
        modg: Modified Group delay Matix
        
        power_product: Power Product Matrix
    """
    ##Global Parameters 
    #global N_frame, N_fft, hop_length, alpha, gamma, N_ny
    
    len_sig = len(sig)
    N_frames = int(len_sig/hop_length + 1)  
    
    ##Normalizing, Pre emphasising, Framing and windowing
    #sig_norm = lb.util.normalize(sig)
    #sig /= sig.max()
    frames = make_windowed_frames(sig)
    
    ##Create y matrix for evaluating group delay function(elementwise Maltiplication)
    y = np.multiply(np.repeat(np.arange(1, N_frame+1, 1).reshape((1,-1)), N_frames, axis = 0), frames)
    
    ##Spectral analysis
    X  = np.ndarray((N_frames, N_fft), dtype = complex)
    Y  = np.ndarray((N_frames, N_fft), dtype = complex)
    
    for i in np.arange(N_frames):
        X[i] = fft(frames[i], n = N_fft)
        Y[i] = fft(y[i], n = N_fft)

    X = X[:, 0:N_ny] + epsilon
    Y = Y[:, 0:N_ny] + epsilon
    
    ##Calculating Cepstrally smoothed spectra
    sm_spectra = np.ndarray((N_frames, N_ny), dtype = complex)
    
    for i in np.arange(N_frames):
        Log_Amp_filterd = dct(signal.medfilt(np.log10(np.abs(X[i])), 7))[0:31]
        sm_spectra[i] = idct(Log_Amp_filterd, n = N_ny)
    
    ##Calculating the modified group delay function and power product
    power_product = X.real*Y.real + X.imag*Y.imag
    temp = power_product/np.power(np.abs(sm_spectra), 2*gamma)
    temp_abs = np.abs(temp)
    modg = (temp/temp_abs)*np.power(temp_abs, alpha)  
    
    if norm_db:
        #return lb.util.normalize(modg.T), lb.util.normalize(power_product.T)
        return lb.amplitude_to_db((modg.T + offset), np.max) ,lb.power_to_db((power_product.T + offset), np.max)
    else:
        return modg.T, power_product.T
    
def mel_modg_spectrogram(sig, norm_db):
    #global sr,N_fft, n_mels, f_max 
    modg, power_product = modg_spectrogram(sig=sig, norm_db=False)
    #Build a mel Filter
    mel_basis = lb.filters.mel(sr=sr, n_fft=N_fft, n_mels=n_mels, fmax=f_max)
    mel_modg  = np.dot(mel_basis, modg)
    mel_power = np.dot(mel_basis, power_product)
    if norm_db:
        #return lb.util.normalize(mel_modg), lb.util.normalize(mel_power)
        return lb.amplitude_to_db((mel_modg + offset), np.max), lb.power_to_db((mel_power + offset), np.max)
    else:
        return mel_modg, mel_power

##Test mode
test =  False

if __name__ == '__main__':
    
    if test:
       
##############################test############################################
######################Audio_signal############################################    
        path_root = 'X:\\Validierungsdaten\\MIREX - Beethoven Op.18 N.5'
        #folder_name = 'test_data'
        file_name = 'bassoon_var5a.wav'
        path = os.path.join(path_root, file_name)
        sig, sr = lb.core.load('D:\\code_dataset\\london_phill_dataset_multi\\oboe\\oboe_Gs4_05_mezzo-forte_normal.mp3')
        #sig, sr = lb.core.load('D:\\code_dataset\\musicnet\\test_data\\1759.wav',offset = 10 ,sr = sr, duration = 1)
########################Test Signal###########################################    
        n = np.arange(3*sr)
        n_1 = np.zeros(3*sr)
        n_2 = np.zeros(3*sr)
        n_1[0:2*sr] = np.arange(2*sr)
        n_2[2*sr:3*sr] = np.arange(sr)
        f_1, f_2, f_3 = 1000, 1000, 4000
        #sig = np.sin(2*np.pi*(1/sr)*n) + 0.0001*np.sin(2*np.pi*(f_3/sr)*n_1)  + 0.25*np.sin(2*np.pi*(f_2/sr)*n) + 0.0001*np.sin(2*np.pi*(f_3/sr)*n_1)
        #sig = np.sin(2*np.pi*(f_1/sr)*n)
##############Modified group delay spectrum and power Spectrum#################
        modg, power_product = modg_spectrogram(sig = sig, norm_db=True)
###############Plotting########################################################
        plt.figure(figsize=(15,15))
##plot MODG Spectrogram
        plt.subplot(3,3,4)
        lb.display.specshow(modg, sr=sr,
                            x_axis='time', y_axis='linear', hop_length = 512)
        plt.colorbar()
        plt.title('Modified group delay Spectrogram')

##plot Power Spectrogram
        plt.subplot(3,3,5)
        lb.display.specshow(power_product, sr=sr,
                            x_axis='time', y_axis='linear', hop_length = 512)
        plt.colorbar()
        plt.title('Power Spectrogram')
    
##Plot STFT Spectrogram
        plt.subplot(3,3,1)
        y_stft = stft_spectrogram(sig, norm_db=True)
        lb.display.specshow(y_stft, sr=sr,
                            x_axis='time', y_axis='linear', hop_length = 512)
        plt.colorbar()
        plt.title('STFT Spectrogram')
##Plot Mel Spectrogram
        plt.subplot(3,3,2)
        y_mel = mel_spectrogram(sig, norm_db=True)
        lb.display.specshow(y_mel, sr=sr, y_axis='mel', x_axis='time', fmax=f_max)
        plt.colorbar()
        plt.title('Mel Spectrogram')
##Plot CQT Spectrogram
#        plt.subplot(3,3,3)
#        y_cqt = np.abs(lb.core.cqt(sig, sr=sr, hop_length=hop_length,
#                                   n_bins= 168, bins_per_octave = 24, window='hamming'))        
#    
#        y_cqt = np.log10(y_cqt) 
#        lb.display.specshow(y_cqt, sr=sr, 
#                            x_axis = 'time', y_axis='cqt_hz', bins_per_octave = 24)    
#        plt.colorbar()
#        plt.title('CQT Spectrogram')  
##Plot Mel-MODG and Power Product Spectrogram
        plt.subplot(3,3,6)
        mel_modg, mel_power = mel_modg_spectrogram(sig, norm_db=True )
        lb.display.specshow(mel_modg, sr=sr, y_axis='mel', x_axis='time', fmax=f_max)
        plt.colorbar()
        plt.title('Mel-MODG Spectrogram')    
    
        plt.subplot(3,3,7)
        lb.display.specshow(mel_power, sr=sr, y_axis='mel', x_axis='time', fmax=f_max)
        plt.colorbar()
        plt.title('Mel-Power Product Spectrogram')  
    
#####################################################################################
#####################################################################################
#y_stft_mean = np.mean(y_stft, axis = 0)