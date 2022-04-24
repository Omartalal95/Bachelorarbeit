# -*- coding: utf-8 -*-
"""
Created on Wed May  8 10:21:50 2019

@author: Omar Elaiashy

This Skript has all the configuration parameters

Instrument's Dictionary: 
    {'1':Piano, '41':Violin, '42':'Viola', '43':'Cello', '61':'Horn',
    '71':'Basson', '72':'Clarient'}  
"""
import os
from six.moves import cPickle as pickle 
import librosa as lb

##Globle Variable für Signalverarbeitung
sr = 44100 #Hz
N_frame = 1024 #Frame of 23.2ms; Length of frame in Samples
hop_length = 512 #Shiffting Window of 11.6ms
N_fft = 1024
N_ny = int(N_fft/2 + 1) #Nyquist frequency in Samples  
epsilon = 1e-3
offset = 10e-3
##Global Parameters for mel Spectrogram
f_max = 11025
n_mels = 128    

##Globale Parameters to control dynamic range of MODGF
alpha = 0.4
gamma = 0.99
pre_emphasis = 0.97

##Global Parameters for pre_dataset
inst_dict = { 1:0, 41:1, 42:2, 43:3, 61:4, 71:5, 72:6} #index in Label Matrix
clip_len_s = 3 # in seconds
clip_len = sr *clip_len_s
N_frames = int(clip_len/hop_length + 1)
frame_len_s = 100e-3 #100ms 
frame_len = frame_len_s*sr
n_frames = int(clip_len_s/frame_len_s)
n_instruments = 7


#if __name__ == '__main__':
#    
#    list_folder = 'D:\\code_dataset\\musicnet\\train_data'
#    dataset_folder = 'D:\\code_dataset\\data\\trainset'
#      
#    audio_IDs = []
#    
#    batch_dict = dict()
#
#    for audio in os.listdir(list_folder):
#    
#        audio_path = os.path.join(list_folder, audio)
#        
#        if not os.path.isdir(audio_path) and not audio.startswith('.'):
#        
#            audio_name = audio.split('.')[0]
#            audio_IDs.append(audio_name)
#        
#            sig, s_r = lb.core.load(audio_path, sr = sr)
#        
#            sig_len = len(sig)
#            clip_len = sr *clip_len_s
#        
#            m = int(sig_len/clip_len)
#            diff = sig_len - m*clip_len
#            
#            if diff/sr >= clip_len_s/2:
#                m = m + 1
#            
#            audio_dict_path = os.path.join(dataset_folder, (audio_name+'.pkl'))
#            with open(audio_dict_path,'rb') as f:
#                audio_dict = pickle.load(f)
#            
#            if audio_dict['batch_size'] == m :
#                
#                batch_dict[audio_name] = m 
#            else:
#                print('Hay, Etwas stimmt nicht')
#                print(audio_dict_path)
#    #to do        
#    total_batches = 0    
#    total_batches_val = 0
#    total_batches_train = 0    
#    audio_IDs_train = []
#    audio_IDs_val = ['2202', '2505', '2398', '2219', '2320', '1893', '1730','1735',
#                     
#                     '1776', '2194', '2300', '2564', '2566', '2588', '2614', '2423',
#                     
#                     '2478', '2486', '2490', '2492', '2343', '2364', '2388', '2393',
#                     
#                     '2678', '1757', '2210', '2212', '2529', '1788', '1835', '1916',
#                     
#                     '1933', '2365', '2451', '2104', '2242', '2560', '1805']
#    
#    
#    save_path_audio_IDs = 'H:\\03_musicnet_modell\\misll\\audio_IDs.pkl'  
#    save_path_batch_dict = 'H:\\03_musicnet_modell\\misll\\batch_dict.pkl'    
#    save_path_audio_IDs_val = 'H:\\03_musicnet_modell\\misll\\audio_IDs_val.pkl'
#    save_path_audio_IDs_train = 'H:\\03_musicnet_modell\\misll\\audio_IDs_train.pkl'
#    save_path_total_batches_val = 'H:\\03_musicnet_modell\\misll\\total_batches_val.pkl'
#    save_path_total_batches_train = 'H:\\03_musicnet_modell\\misll\\total_batches_train.pkl'
#    save_path_total_batches = 'H:\\03_musicnet_modell\\misll\\total_batches.pkl'
#    
#       
#    for ID in audio_IDs_val:
#        
#        total_batches_val += batch_dict[ID] 
#    
#    for ID in audio_IDs:
#        total_batches += batch_dict[ID]      
#    
#    for ID in audio_IDs:
#        
#        if not (ID in audio_IDs_val):
#            
#            audio_IDs_train.append(ID)
#            total_batches_train += batch_dict[ID] 
#    
#    with open(save_path_audio_IDs, 'wb') as f:
#        pickle.dump(audio_IDs, f, pickle.HIGHEST_PROTOCOL)    
#
#    with open(save_path_batch_dict, 'wb') as f:
#        pickle.dump(batch_dict, f, pickle.HIGHEST_PROTOCOL) 
#    
#    with open(save_path_audio_IDs_val, 'wb') as f:
#        pickle.dump(audio_IDs_val, f, pickle.HIGHEST_PROTOCOL) 
#    
#    with open(save_path_audio_IDs_train, 'wb') as f:
#        pickle.dump(audio_IDs_train, f, pickle.HIGHEST_PROTOCOL)  
#        
#    with open(save_path_total_batches_val, 'wb') as f:
#        pickle.dump(total_batches_val, f, pickle.HIGHEST_PROTOCOL) 
#         
#    with open(save_path_total_batches, 'wb') as f:
#        pickle.dump(total_batches, f, pickle.HIGHEST_PROTOCOL) 
#        
#    with open(save_path_total_batches_train, 'wb') as f:
#        pickle.dump(total_batches_train, f, pickle.HIGHEST_PROTOCOL)    