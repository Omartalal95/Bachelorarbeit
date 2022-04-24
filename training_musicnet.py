# -*- coding: utf-8 -*-
"""
Created on Mon May 27 13:23:54 2019

@author: Omar Elaiashy
"""

import numpy
import time
import os
from six.moves import cPickle as pickle
import keras.backend as K
from keras.optimizers import SGD
from datagenerator import DataGenerator
from modell import classifier
from keras.callbacks import History, EarlyStopping, ModelCheckpoint
from keras.callbacks import TensorBoard
import tensorflow as tf
from tensorflow.python.client import device_lib 
from keras.backend.tensorflow_backend import set_session
from tensorflow.python.client import device_lib
from keras import backend as K

# Parameters
train_path = 'D:\\Omar_Elaiashy\\03_musicnet_modell\\data\\trainset' # to do 
valid_path = path = 'D:\\Omar_Elaiashy\\03_musicnet_modell\\data\\trainset' # to do 

audio_IDs_train_path = 'D:\\Omar_Elaiashy\\03_musicnet_modell\\misll\\audio_IDs_train.pkl' # to do   
       
with open(audio_IDs_train_path,'rb') as f: 
    audio_IDs_train = pickle.load(f)

audio_IDs_valid_path = 'D:\\Omar_Elaiashy\\03_musicnet_modell\\misll\\audio_IDs_val.pkl' # to do   
      
with open(audio_IDs_valid_path,'rb') as f:
    audio_IDs_val = pickle.load(f)

batch_dict_path = 'D:\\Omar_Elaiashy\\03_musicnet_modell\\misll\\batch_dict.pkl' # to do  
       
with open(batch_dict_path,'rb') as f:
    batch_dict = pickle.load(f)

sig_processing = 'stft+modg'
batch_size = 100


#todo
total_batches_train_path = 'D:\\Omar_Elaiashy\\03_musicnet_modell\\misll\\total_batches_train.pkl'

with open(total_batches_train_path,'rb') as f:
    total_batches_train = pickle.load(f)

total_batches_val_path = 'D:\\Omar_Elaiashy\\03_musicnet_modell\\misll\\total_batches_val.pkl'

with open(total_batches_val_path,'rb') as f:
    total_batches_val = pickle.load(f)

# Generators
#path #todo
training_generator = DataGenerator(path = train_path, 
                                   audio_IDs=audio_IDs_train, 
                                   batch_dict=batch_dict,
                                   sig_processing = sig_processing,
                                   total_batches=total_batches_train,
                                   dim=(513, 259),
                                   dim_label=(30, 7),
                                   n_channels=2,
                                   batch_size = batch_size)
validation_generator = DataGenerator(path = valid_path, 
                                   audio_IDs=audio_IDs_val, 
                                   batch_dict=batch_dict,
                                   sig_processing = sig_processing,
                                   total_batches=total_batches_val,
                                   dim=(513, 259),
                                   dim_label=(30, 7),
                                   n_channels=2,
                                   batch_size = 32)

# Design model
model = classifier()

# Optimizer
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9)

model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['binary_accuracy'])
# Callbacks
callbacks = []
#
#save_path_tensorboard = 'H:\\04_musicnet_Results\\tensorboard' #to do
#
#if K.backend() == 'tensorflow':
#    tensorboard = TensorBoard(log_dir=os.path.join(save_path_tensorboard, 'logs'),
#                              histogram_freq=10, write_graph=True)
#    callbacks.append(tensorboard)

#todo
save_path_modelcheckpoint="D:\\Omar_Elaiashy\\04_musicnet_Results\\modell\\best_model_stft_modg.hdf5" 

hist = History();
earlystop = EarlyStopping(monitor='val_binary_acc', min_delta=0.01, restore_best_weights=True, patience= 10, verbose=1 )
checkpoint = ModelCheckpoint(save_path_modelcheckpoint, monitor='val_binary_acc', verbose=1, save_best_only=True, mode='max')

callbacks.append(hist)
callbacks.append(earlystop)
callbacks.append(checkpoint)

# Train model on dataset
epochs = 30

#init_op = tf.global_variables_initializer()
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
#sess.run(init_op)
K.set_session(session)

#Dont use use _multiprocessing=True I dont know why it doesnt work
#Sometimes it doesnt work ,when the insilisation of Batch normalization dot work  
start = time.time()                    
#hi = model.fit_generator(generator=training_generator,
#                    epochs=epochs, 
#                    verbose=1,max_queue_size=1) #That works
model.fit_generator(generator=training_generator,
                    epochs=epochs, 
                    verbose=1,
                    validation_data=validation_generator,
                    callbacks=callbacks, 
                    max_queue_size=1)

end = time.time()
print(end - start)
# Save History
#todo
save_path_history = 'D:\\Omar_Elaiashy\\04_musicnet_Results\\history\\history_stft_modg.pkl'
with open(save_path_history, 'wb') as f:
    pickle.dump(hist, f, pickle.HIGHEST_PROTOCOL)


# Clear session
K.clear_session()

def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss')
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss')
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


plot_history(hist)
