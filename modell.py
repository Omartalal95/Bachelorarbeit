# -*- coding: utf-8 -*-
"""
Created on Fri May 24 15:58:13 2019

@author: Omar Elaiashy
"""

from keras.models import Model
from keras.regularizers import l2
from keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout 


def classifier(input_shape = (513,259,2) ,reg = 0.0001, classes = 7):
    
    spect_input = Input(shape = input_shape)
    
    # Block 1
    x = Conv2D(16, (11, 3), padding='same', data_format="channels_last",
               name='block1_conv1', kernel_regularizer=l2(reg))(spect_input)
    x = BatchNormalization(name='block1_bn1')(x)
    x = Activation('relu', name='block1_Ac1')(x)
    
    x = Conv2D(16, (11, 3), padding='same', data_format="channels_last", 
               name='block1_conv2', kernel_regularizer=l2(reg))(x)
    #x = BatchNormalization(name='block1_bn2')(x)
    x = Activation('relu', name='block1_Ac2')(x)
    
    x = MaxPooling2D((3, 2), strides=(3, 2), name='block1_pool')(x)
    
    # Block 2
    x = Conv2D(32, (11, 5), padding='same', data_format="channels_last", 
               name='block2_conv1', kernel_regularizer=l2(reg))(x)
    x = BatchNormalization(name='block2_bn1')(x)
    x = Activation('relu', name='block2_Ac1')(x)
    
    x = Conv2D(32, (11, 5), padding='same', data_format="channels_last", 
               name='block2_conv2', kernel_regularizer=l2(reg))(x)
    x = BatchNormalization(name='block2_bn2')(x)
    x = Activation('relu', name='block2_Ac2')(x)
    
    x = MaxPooling2D((3, 2), strides=(3, 2), name='block2_pool')(x)
    
    # Block 3
    x = Conv2D(64, (11, 7), padding='same', data_format="channels_last", 
               name='block3_conv1', kernel_regularizer=l2(reg))(x)
    x = BatchNormalization(name='block3_bn1')(x)
    x = Activation('relu', name='block3_Ac1')(x)
##comment    
    x = Conv2D(64, (11, 7), padding='same', data_format="channels_last",
               name='block3_conv2', kernel_regularizer=l2(reg))(x)
    x = BatchNormalization(name='block3_bn2')(x)
    x = Activation('relu', name='block3_Ac2')(x)
    
    x = Conv2D(64, (11, 7), padding='same', data_format="channels_last",
               name='block3_conv3', kernel_regularizer=l2(reg))(x)
    x = BatchNormalization(name='block3_bn3')(x)
    x = Activation('relu', name='block3_Ac3')(x)
    
    x = MaxPooling2D((3, 2), strides=(3, 2), name='block3_pool')(x)
    
    # Block 4
    x = Conv2D(128, (11, 9), padding='same', data_format="channels_last",
               name='block4_conv1', kernel_regularizer=l2(reg))(x)
    x = BatchNormalization(name='block4_bn1')(x)
    x = Activation('relu', name='block4_Ac1')(x)
##comment    
    x = Conv2D(128, (11, 9), padding='same', data_format="channels_last", 
               name='block4_conv2',kernel_regularizer=l2(reg))(x)
    #x = BatchNormalization(name='block4_bn2')(x)
    x = Activation('relu', name='block4_Ac2')(x)
    
    x = Conv2D(128, (11, 9), padding='same', data_format="channels_last", 
               name='block4_conv3',kernel_regularizer=l2(reg))(x)
    x = BatchNormalization(name='block4_bn3')(x)
    x = Activation('relu', name='block4_Ac3')(x)
    
    x = MaxPooling2D((3, 1), strides=(3, 1), name='block4_pool')(x)
    
    #Matching Layer 4069
    x = Conv2D(2048, (6, 3), name='matching_layer', data_format="channels_last",
               kernel_regularizer=l2(reg))(x)
    x = BatchNormalization(name='matching_layer_bn')(x)
    x = Activation('relu', name='matching_layer_Ac')(x)
##comment    
    # fully-connected layers 4069
    x = Conv2D(1024, (1, 1), activation='relu', padding='same', 
               data_format="channels_last", name='fc1', 
               kernel_regularizer=l2(reg))(x)
    
    x = Dropout(0.50)(x)
    
    x = Conv2D(512, (1, 1), activation='relu', padding='same', 
               data_format="channels_last", name='fc2', 
               kernel_regularizer=l2(reg))(x)

    x = Dropout(0.30)(x)
    
    x = Conv2D(256, (1, 1), activation='relu', padding='same', 
               data_format="channels_last", name='fc3', 
               kernel_regularizer=l2(reg))(x)

    x = Dropout(0.20)(x)
    
    x = Conv2D(128, (1, 1), activation='relu', padding='same', 
               data_format="channels_last", name='fc4', 
               kernel_regularizer=l2(reg))(x)

    x = Dropout(0.15)(x)
    
    #Output layer
    x = Conv2D(classes, (1, 1), kernel_initializer='he_normal', 
               activation='sigmoid', padding='valid', 
               strides=(1, 1), kernel_regularizer=l2(reg))(x)
    
    return Model(spect_input, x)
#    model =  Model(spect_input, x)
#    print(model.summary())
    #print(model.input_shape)
    #print(model.output_shape)

    
    


    
