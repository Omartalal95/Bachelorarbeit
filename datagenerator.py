# -*- coding: utf-8 -*-
"""
Created on Wed May 15 17:52:15 2019

based on tutorial by Afshine Amidi and Shervine Amidi of Stanford University
  ''A detailed example of how to use data generators with Keras''

@author: Omar Elaiashy
"""

import numpy as np
import keras
from six.moves import cPickle as pickle #for performance
import os

class DataGenerator(keras.utils.Sequence):
     'Generates data for Keras'
     def __init__(self, path, audio_IDs, batch_dict, 
                  sig_processing, total_batches, batch_size = 128, 
                  dim=(513, 259), dim_label=(30, 7) ,n_channels=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.dim_label = dim_label
        self.batch_size = batch_size
        self.path = path
        self.audio_IDs = audio_IDs
        self.batch_dict = batch_dict
        self.sig_processing = sig_processing
        self.total_batches = total_batches
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.audio_pointer = 0
        self.position_pointer = 0
        self.on_epoch_end()
     
     def __len__(self):
        return int(np.ceil(self.total_batches / self.batch_size))  #floor()
    
     def __getitem__(self,x ):
        'Generate one batch of data'
        X, y = self.__data_generation()
        return X, y
    
     def on_epoch_end(self):
        
        self.audio_pointer = 0 #Reset Audio_pointer
        self.position_pointer = 0 #Reset Position_pointer
        
        if self.shuffle:
            np.random.shuffle(self.audio_IDs)
            
     def fetch_data(self, dataset_path,start_pointer, end_pointer):
        
        #print(dataset_path)
        with open(dataset_path, 'rb') as f:
            audio_dict = pickle.load(f)
        return audio_dict[self.sig_processing][start_pointer:end_pointer], audio_dict['label'][start_pointer:end_pointer]
    
     def __data_generation(self):
        
        if self.audio_pointer >= len(self.audio_IDs):
            self.audio_pointer = 0
            self.position_pointer = 0
        if (self.audio_pointer + 1) == len(self.audio_IDs) and  (self.position_pointer + self.batch_size) > self.batch_dict[self.audio_IDs[self.audio_pointer]]:
            #print('Halloo')
            if self.position_pointer == self.batch_dict[self.audio_IDs[self.audio_pointer]]:
                #print('Halloo')
                self.audio_pointer = 0
                self.position_pointer = 0 
        
        if (self.position_pointer + self.batch_size) <= self.batch_dict[self.audio_IDs[self.audio_pointer]]:
            
            dataset_path = os.path.join(self.path, str(self.audio_IDs[self.audio_pointer])+'.pkl')
            start_pointer = self.position_pointer
            end_pointer = self.position_pointer + self.batch_size
            #print('***<=***')
            #print(start_pointer,end_pointer,self.audio_IDs[self.audio_pointer],batch_dict[self.audio_IDs[self.audio_pointer]])
            X, y = self.fetch_data(dataset_path, start_pointer, end_pointer)
            
            if (self.position_pointer + self.batch_size) == self.batch_dict[self.audio_IDs[self.audio_pointer]]:
                
                self.position_pointer = 0
                self.audio_pointer += 1
                #print('******==******')
            else:
                
                self.position_pointer = self.position_pointer + self.batch_size
                #print('******<******')
        elif (self.position_pointer + self.batch_size) > self.batch_dict[self.audio_IDs[self.audio_pointer]]: 
            #print('***>***')
            if (self.audio_pointer + 1) == len(self.audio_IDs):
                
                dataset_path = os.path.join(self.path, str(self.audio_IDs[self.audio_pointer])+'.pkl')
                #check start_pointer == self.batch_dict[self.audio_IDs[self.audio_pointer]]
                start_pointer = self.position_pointer
                end_pointer = self.batch_dict[self.audio_IDs[self.audio_pointer]]
                
                X, y = self.fetch_data(dataset_path, start_pointer, end_pointer)
                
                self.audio_pointer = 0
                self.position_pointer = 0
            
            else:
                
                diffr = self.batch_dict[self.audio_IDs[self.audio_pointer]] - self.position_pointer
                temp = 0
                temp_mat_x = np.zeros(shape=(1, *self.dim, self.n_channels))
                temp_mat_y = np.zeros(shape=(1, 1, *self.dim_label))
                #print(temp_mat_x.shape)
                counter = self.audio_pointer
                
                for i in np.arange(self.audio_pointer, len(self.audio_IDs)):
                    counter = i
                    if i == self.audio_pointer:
                        temp = diffr
                    else:
                        temp += self.batch_dict[self.audio_IDs[i]]
                    if temp >= self.batch_size:
                        break
                            
                for i in np.arange(self.audio_pointer,counter + 1):
                                
                    start_pointer = 0
                    end_pointer = self.batch_dict[self.audio_IDs[i]]
                    
                    if i == self.audio_pointer:
                        start_pointer = self.position_pointer
                
                    if i == counter:
                        #print('+++++++++')
                        #print(temp_mat_x.shape, self.batch_size - temp_mat_x.shape[0],start_pointer)
                        #print('+++++++++')
                        if (self.batch_size - temp_mat_x.shape[0]) < self.batch_dict[self.audio_IDs[i]]: 
                            end_pointer = self.batch_size - temp_mat_x.shape[0] + 1
                            self.audio_pointer = counter
                            self.position_pointer = end_pointer
                        else:   
                            self.audio_pointer = counter + 1
                            self.position_pointer = 0
                            
                    dataset_path = os.path.join(self.path, str(self.audio_IDs[i])+'.pkl')
                    fetched_data_x, fetched_data_y = self.fetch_data(dataset_path, start_pointer, end_pointer)
                    #print(fetched_data_y.shape)
                    #print(temp_mat_y.shape)
                    temp_mat_x = np.vstack((temp_mat_x, fetched_data_x))
                    temp_mat_y = np.vstack((temp_mat_y, fetched_data_y))
                    
                X = temp_mat_x[1:]
                y = temp_mat_y[1:]
                      
        return X, y


#path = os.path.join('D:/try', 'trainset')
#audio_IDs = [1759, 1819, 1728, 1729]
#batch_dict = {1728:84, 1729:148, 1759:65, 1819:59} 
#sig_processing = 'stft+modg'
#total_batches = 356
#
#Datagenerator = DataGenerator(path = path, audio_IDs=audio_IDs, batch_dict=batch_dict
#                            ,sig_processing = sig_processing, total_batches=total_batches
#                            ,batch_size = 128)
#
#for i in range(11):
#    
#    print(Datagenerator.audio_IDs)
#    
#    X, y = Datagenerator.__getitem__()  
#   
##    print('***********')
##    print(X.shape)
##    print(y.shape)
##    if i == 1:
##        print(X[1])
##        print(y[1])
##        print('***********')
##        print(X[-1])
##        print(y[-1])
##    print('***********')    