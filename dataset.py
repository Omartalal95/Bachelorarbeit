# -*- coding: utf-8 -*-
"""
Created on Wed May  8 09:37:36 2019

@author: Omar Elaiashy

Instrument's Dictionary: 
    {'1':Piano, '41':Violin, '42':'Viola', '43':'Cello', '61':'Horn',
    '71':'Basson', '72':'Clarient'}                 
"""
from config import*
import os
from six.moves import cPickle as pickle #for performance
import signal_processing
import numpy as np
import pandas as pd
import librosa as lb
import librosa.display
import time
#import math
#import matplotlib.pyplot as plt

def make_clips(sig, sr, clip_len_s):
    
    sig_len = len(sig)
    clip_len = sr *clip_len_s
    m = int(sig_len/clip_len)
    diff = sig_len - m*clip_len

    if diff/sr >= clip_len_s/2:
        m = m + 1
        size_sig = m*clip_len_s*sr
    else:
        size_sig = m*clip_len_s*sr
    
    sig = lb.util.fix_length(sig, size=size_sig)
    
    clips = np.ndarray(shape=(m, clip_len))
    
    for i in np.arange(m):
        clips[i] = sig[i*clip_len:(i+1)*clip_len]
    
    return clips, m  

def concatenate_channel_dim(matrix_1, matrix_2):
    
    matrix_1 = np.expand_dims(matrix_1, axis = 3)
    matrix_2 = np.expand_dims(matrix_2, axis = 3)
    
    return np.concatenate((matrix_1, matrix_2), 3)

def compute_signal_processing(clips):
    
    stft_matrix = np.ndarray(shape=(clips.shape[0], N_ny, N_frames))
    modg_matrix = np.ndarray(shape=(clips.shape[0], N_ny, N_frames))
    power_matrix = np.ndarray(shape=(clips.shape[0], N_ny, N_frames))
    mel_matrix = np.ndarray(shape=(clips.shape[0], n_mels, N_frames))
    modg_mel_matrix = np.ndarray(shape=(clips.shape[0], n_mels, N_frames))
    power_mel_matrix = np.ndarray(shape=(clips.shape[0], n_mels, N_frames)) 
    
    for i in np.arange(clips.shape[0]):
        stft_matrix[i] = signal_processing.stft_spectrogram(clips[i],True)
        modg_matrix[i], power_matrix[i] = signal_processing.modg_spectrogram(clips[i], True)
        mel_matrix[i] = signal_processing.mel_spectrogram(clips[i],True)
        modg_mel_matrix[i], power_mel_matrix[i] = signal_processing.mel_modg_spectrogram(clips[i],True)
        
    return stft_matrix, modg_matrix, power_matrix, mel_matrix, modg_mel_matrix, power_mel_matrix    

def compute_label(path, m):
    
    Label_df = pd.read_csv(filepath_or_buffer = path) 
    label_matrix = np.zeros(shape=(m, 1, n_frames,n_instruments)) #to do
    #label_matrix = np.zeros(shape=(m, n_instruments, n_frames)) #m = clips.shape[0] 

    for j in np.arange(m): 
   
        for i in np.arange(n_frames):
    
            start_time_frame = i*frame_len + j*clip_len 
            end_time_frame = (i+1)*frame_len + j*clip_len
        
            Label_df_sub = Label_df.copy()
            Label_df_sub['con_1'] = Label_df['end_time'] - start_time_frame
            Label_df_sub['con_2'] = end_time_frame - Label_df['start_time']
        
            con_1 = Label_df_sub['con_1'] > 0
            con_2 = Label_df_sub['con_2'] > 0

            df =Label_df_sub[con_1 & con_2]
        
            for index, row in df.iterrows():
                #start_time_cell = row['start_time']
                #end_time_cell = row['end_time']
                inst = row['instrument']
                if inst in [1, 41, 42, 43, 61, 71, 72]:
                    label_matrix[j,0,i,inst_dict[inst]] = 1
                    #label_matrix[j,inst_dict[inst],i] = 1
            
            del df
            del Label_df_sub            
    return label_matrix

def audio_processing(sig, csv_path):
    
    audio_dict = dict()
    
    clips, m = make_clips(sig, sr, clip_len_s)
    stft_matrix, modg_matrix, power_matrix, mel_matrix, modg_mel_matrix, power_mel_matrix = compute_signal_processing(clips)
    label_matrix = compute_label(csv_path, m)
    
    audio_dict['stft+modg'] = concatenate_channel_dim(stft_matrix, modg_matrix)
    audio_dict['stft+power'] = concatenate_channel_dim(stft_matrix, power_matrix)
    audio_dict['mel+mel-modg'] = concatenate_channel_dim(mel_matrix, modg_mel_matrix)
    audio_dict['mel+mel-power'] = concatenate_channel_dim(mel_matrix, power_mel_matrix)
    audio_dict['label'] = label_matrix
    audio_dict['batch_size'] = m
    
    return audio_dict
    #return m

#audio_path = os.path.join(data_folder, audio)
#            
#if not os.path.isdir(audio_path) and not audio.startswith('.'):
#                
#    audio_name = audio.split('.')[0]
#    csv_path = os.path.join(label_folder,(audio_name+'.csv'))    
#    save_path = os.path.join(save_folder, (audio_name+'.pkl'))  
#                   
#    sig, s_r = lb.core.load('D:\\code_dataset\\musicnet\\train_data\\1751.wav', sr = sr)
#    sig /= sig.max()
#    #sig = [0 if math.isnan(x) else x for x in sig]                
#    audio_dict = audio_processing(sig, csv_path = 'D:\\code_dataset\\musicnet\\train_labels\\1751.csv')
#                    #m.append(audio_processing(sig, csv_path = csv_path))
#                    #m_t.append(audio_processing(sig, csv_path = csv_path))
#    
#    
#    with open('D:\\code_dataset\\1751.pkl' , 'wb') as f:
#        pickle.dump(audio_dict, f, pickle.HIGHEST_PROTOCOL)



#Save Data according to the Structure of Data Generator
if __name__ == '__main__':
    
    start = time.time()
    
    root = 'D:\\code_dataset\\musicnet'
    save_root = 'D:\\code_dataset\\data'
    sets = ['train_set','test_set']
    
    #m = []
    #m_t = []
    for st in sets:
        
        if st == 'train_set':
            audio_folder = 'train_data'
            csv_folder = 'train_labels'
            set_folder = 'trainset'
        else:
            audio_folder = 'test_data'
            csv_folder = 'test_labels'
            set_folder = 'testset'
            
        data_folder  = os.path.join(root, audio_folder)
        print(data_folder)
        label_folder = os.path.join(root, csv_folder)
        print(label_folder)
        save_folder   = os.path.join(save_root,set_folder) 
        print(save_folder)
        
        for audio in os.listdir(data_folder):
            
            audio_path = os.path.join(data_folder, audio)
            
            if not os.path.isdir(audio_path) and not audio.startswith('.'):
                
                audio_name = audio.split('.')[0]
                csv_path = os.path.join(label_folder,(audio_name+'.csv'))    
                save_path = os.path.join(save_folder, (audio_name+'.pkl'))  
               
                sig, s_r = lb.core.load(audio_path, sr = sr)
                
                if sig.max() != 0:
                    sig /= sig.max()
                
                audio_dict = audio_processing(sig, csv_path = csv_path)
                #m.append(audio_processing(sig, csv_path = csv_path))
                #m_t.append(audio_processing(sig, csv_path = csv_path))
                with open(save_path, 'wb') as f:
                    pickle.dump(audio_dict, f, pickle.HIGHEST_PROTOCOL)
    
    end = time.time()
    print(end - start)
    sig.max()