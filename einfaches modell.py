# -*- coding: utf-8 -*-

# Imports

#General
import numpy as np
import itertools
from six.moves import cPickle as pickle #for performance

# System
import os, fnmatch

# Data
import pandas as pd

# Visualization
import seaborn 
import matplotlib.pyplot as plt
from IPython.core.display import HTML, display, Image

#Signal Processing
from scipy.fftpack import fft
from scipy.fftpack import dct
from scipy.fftpack import idct
from scipy import signal

# Machine Learning
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import recall_score, precision_score, accuracy_score
from sklearn.metrics import confusion_matrix, f1_score, classification_report


# Deep Learning
import tensorflow as tf
from tensorflow.python.client import device_lib 
from keras.backend.tensorflow_backend import set_session
from tensorflow.python.client import device_lib
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, merge
from keras.layers.normalization import BatchNormalization
from keras.callbacks import History, EarlyStopping, ModelCheckpoint
from keras.models import load_model


# Random Seed
from tensorflow import set_random_seed
from numpy.random import seed
seed(0)
set_random_seed(0)

# Audio
import librosa.display, librosa
from librosa.util import normalize as normalize
import IPython.display as ipd

# Configurations
#path='./audio/london_phill_dataset_multi/'
path = 'H:\\01_einfaches modell\\1.01_london_phill_dataset_multi'
# Display CPUs and GPUs
print(device_lib.list_local_devices())

# Signal Processing Parameters
fs = 44100         # Sampling Frequency
n_fft = 2048       # length of the FFT window
hop_length = 512   # Number of samples between successive frames
N_ny = int(n_fft/2 + 1) #Nyquist frequency in Samples  
epsilon = 1e-3
alpha = 0.4
gamma = 0.99
pre_emphasis = 0.97
f_max = 11025
n_mels = 128
N_fft = 2048
# Machine Learning Parameters
testset_size = 0.25 #Percentage of data for Testing
sr = fs
#Find Audio Files
#files = []
#labels =[]
#duration = []
#classes=['flute','sax','oboe', 'cello','trumpet','viola']
#for root, dirnames, filenames in os.walk(path):
#    for i, filename in enumerate(fnmatch.filter(filenames, '*.mp3')):
#        files.append(os.path.join(root, filename))
#        for name in classes:
#            if fnmatch.fnmatchcase(filename, '*'+name+'*'):
#                labels.append(name)
#                break
#        else:
#            labels.append('other')
#        print ("Get %d = %s"%(i+1, filename))
#        try:
#            y, sr = librosa.load(files[i], sr=fs)
#            if len(y) < 2:
#                print("Error loading %s" % filename)
#                continue
#            #y/=y.max() #Normalize
#            yt, index = librosa.effects.trim(y,top_db=60) #Trim
#            duration.append(librosa.get_duration(yt, sr=fs))
#        except Exception as e:
#            print("Error loading %s. Error: %s" % (filename,e))
#
#with open('H:\\01_einfaches modell\\1.04_misll\\files.pkl', 'wb') as f:
#    pickle.dump(files, f, pickle.HIGHEST_PROTOCOL)
#    
#with open('H:\\01_einfaches modell\\1.04_misll\\labels.pkl', 'wb') as f:
#    pickle.dump(labels, f, pickle.HIGHEST_PROTOCOL)


 
with open('H:\\01_einfaches modell\\1.04_misll\\files.pkl', 'rb') as f:
    files = pickle.load(f)

with open('H:\\01_einfaches modell\\1.04_misll\\labels.pkl', 'rb') as f:
    labels = pickle.load(f)    

#print("found %d audio files in %s"%(len(files),path))
#
#print("Max. Duration:", max(duration))
#print("Min. Duration:", min(duration))
#print("Average Duration:", np.mean(duration))

# Load audio files, trim silence and calculate duration
#duration = []
#for i,f in enumerate(files):
#    print ("Get %d  %s"%(i+1, f))
#    try:
#        y, sr = librosa.load(f, sr=fs)
#        if len(y) < 2:
#            print("Error loading %s" % f)
#            continue
#        #y/=y.max() #Normalize
#        yt, index = librosa.effects.trim(y,top_db=60) #Trim
#        duration.append(librosa.get_duration(yt, sr=fs))
#    except Exception as e:
#        print("Error loading %s. Error: %s" % (f,e))
#        
#print("Calculated %d Durations"%len(duration))
#
#durationDist = pd.Series(np.array(duration))
#plt.figure()
#durationDist.plot.hist(grid=True, bins=40, rwidth=0.8,
#                   color='#607c8e')
#plt.title('Duration Distribution')
#plt.xlabel('Duration [s]')
#plt.ylabel('Counts')
#plt.grid(axis='y', alpha=0.75)
#print("Duration average:",np.mean(duration))

# STFT Example
#y, sr = librosa.load(files[10], sr=fs, duration=1)
#y/=y.max() #Normalize
#duration_in_samples=librosa.time_to_samples(1, sr=fs)
#y_pad = librosa.util.fix_length(y, duration_in_samples) #Pad to 1s if smaller
#y_stft=librosa.core.stft(y_pad, n_fft=n_fft, hop_length=hop_length)
#y_spec=librosa.amplitude_to_db(abs(y_stft), np.max)
#plt.figure(figsize=(14,8))
#plt.title("Short-Time Fourier Transform Spectogram \n %s"%files[0])
#librosa.display.specshow(y_spec,sr=fs,y_axis='log', x_axis='time')
#plt.colorbar(format='%+2.0f dB');
#print("Spectogram Array Shape:",y_spec.shape)

#Encode Labels
labelencoder = LabelEncoder()
labelencoder.fit(labels)
print(len(labelencoder.classes_), "classes:", ", ".join(list(labelencoder.classes_)))
classes_num = labelencoder.transform(labels)
#
##OneHotEncoding
encoder=OneHotEncoder(sparse=False, categories="auto")
onehot_labels=encoder.fit_transform(classes_num.reshape(len(classes_num),1))

## Create Train and Test Sets
splitter = StratifiedShuffleSplit(n_splits=1, test_size=testset_size, random_state=0)
splits = splitter.split(files, onehot_labels)
files_arr=np.array(files)

for train_index, test_index in splits:
    train_set_files = files_arr[train_index]
    test_set_files = files_arr[test_index]
    train_classes = onehot_labels[train_index]
    test_classes = onehot_labels[test_index]

#with open('H:\\01_einfaches modell\\1.04_misll\\train_set_files.pkl', 'wb') as f:
#    pickle.dump(train_set_files, f, pickle.HIGHEST_PROTOCOL)
#
#with open('H:\\01_einfaches modell\\1.04_misll\\test_set_files.pkl', 'wb') as f:
#    pickle.dump(test_set_files, f, pickle.HIGHEST_PROTOCOL)
#
#with open('H:\\01_einfaches modell\\1.04_misll\\train_classes.pkl', 'wb') as f:
#    pickle.dump(train_classes, f, pickle.HIGHEST_PROTOCOL)    
#
#with open('H:\\01_einfaches modell\\1.04_misll\\test_classes.pkl', 'wb') as f:
#    pickle.dump(test_classes, f, pickle.HIGHEST_PROTOCOL)
#    
#with open('H:\\01_einfaches modell\\1.04_misll\\classes_num.pkl', 'wb') as f:
#    pickle.dump(classes_num, f, pickle.HIGHEST_PROTOCOL)  
#with open('H:\\01_einfaches modell\\1.04_misll\\classes_num.pkl', 'wb') as f:
#    pickle.dump(classes_num, f, pickle.HIGHEST_PROTOCOL)  
#    
###################################################################################
#with open('H:\\01_einfaches modell\\1.04_misll\\train_set_files.pkl', 'rb') as f:
#    train_set_files = pickle.load(f)
#
#with open('H:\\01_einfaches modell\\1.04_misll\\test_set_files.pkl', 'rb') as f:
#    test_set_files = pickle.load(f)
#    
#with open('H:\\01_einfaches modell\\1.04_misll\\train_classes.pkl', 'rb') as f:
#    train_classes = pickle.load(f)
#
#with open('H:\\01_einfaches modell\\1.04_misll\\test_classes.pkl', 'rb') as f:
#    test_classes = pickle.load(f)
#
#with open('H:\\01_einfaches modell\\1.04_misll\\classes_num.pkl', 'rb') as f:
#    classes_num = pickle.load(f)
###################################################################################    
## CNN Model
model = Sequential()

conv_filters =  16  # number of convolution filters

# Layer 1
#model.add(Convolution2D(conv_filters, 3,input_shape=(1025, 87, 2)))
model.add(Convolution2D(conv_filters, 3,input_shape=(128, 87, 2)))
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Dropout(0.30)) 

# Layer 2
model.add(Convolution2D(32, 3))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.30))

# Layer 3
model.add(Convolution2D(64, 3))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.30))

# Flatten
model.add(Flatten()) 


# Full layer
model.add(Dense(32, activation='sigmoid')) 

# Output layer
model.add(Dense(6,activation='softmax'))

model.summary()

# Loss Function 
loss = 'categorical_crossentropy' 

# Optimizer = Gradient Descent
optimizer = 'sgd' 

# Compile
model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

#def make_windowed_frames(sig):
#   
#    ##Pre emphasising the Signal
#    emp_sig = np.append(sig[0], sig[1:] - pre_emphasis*sig[:-1])
#    
#    ##Framing the Signal
#    len_sig = len(sig)
#    N_frames = int(len_sig/hop_length + 1)   #Number of Frames 
#    frames = np.ndarray((N_frames, n_fft), dtype= float)
#    
#    ##Windowing the Signal to decreace leakage effect
#    w = signal.hamming(n_fft).reshape((1,-1))
#    
#    for i in np.arange(N_frames):
#        for j in np.arange(n_fft):
#            if (i*hop_length + j) < len_sig:
#                frames[i][j] = emp_sig[i*hop_length + j]*w[0, j]
#            else:
#                frames[i][j] = 0 
#    return frames




def modg_spectrogram(sig, typ = 'modg'):

    len_sig = len(sig)
    N_frames = int(len_sig/hop_length + 1)  
    
        ##Pre emphasising the Signal
    emp_sig = np.append(sig[0], sig[1:] - pre_emphasis*sig[:-1])
    
    ##Framing the Signal
    len_sig = len(sig)
    N_frames = int(len_sig/hop_length + 1)   #Number of Frames 
    frames = np.ndarray((N_frames, n_fft), dtype= float)
    
    ##Windowing the Signal to decreace leakage effect
    w = signal.hamming(n_fft).reshape((1,-1))
    
    for i in np.arange(N_frames):
        for j in np.arange(n_fft):
            if (i*hop_length + j) < len_sig:
                frames[i][j] = emp_sig[i*hop_length + j]*w[0, j]
            else:
                frames[i][j] = 0 
    
    
    ##Create y matrix for evaluating group delay function(elementwise Maltiplication)
    y = np.multiply(np.repeat(np.arange(1, n_fft+1, 1).reshape((1,-1)), N_frames, axis = 0), frames)
    
    ##Spectral analysis
    X  = np.ndarray((N_frames, n_fft), dtype = complex)
    Y  = np.ndarray((N_frames, n_fft), dtype = complex)
    
    for i in np.arange(N_frames):
        X[i] = fft(frames[i], n = n_fft)
        Y[i] = fft(y[i], n = n_fft)

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
    
    if typ == 'modg':
        return modg.T
    elif typ == 'power':
        return power_product.T
    elif typ == 'mel_modg': 
            mel_basis = librosa.filters.mel(sr=sr, n_fft=N_fft, n_mels=n_mels, fmax=f_max)
            return np.dot(mel_basis, modg.T)
    elif typ == 'mel_power':
        mel_basis = librosa.filters.mel(sr=sr, n_fft=N_fft, n_mels=n_mels, fmax=f_max)
        return np.dot(mel_basis, power_product.T)



# modg Example
#y, sr = librosa.load(files[10], sr=fs, duration=1)
#y/=y.max() #Normalize
#duration_in_samples=librosa.time_to_samples(1, sr=fs)
#y_pad = librosa.util.fix_length(y, duration_in_samples) #Pad to 1s if smaller
#y_modg = modg_spectrogram(y_pad, typ = 'mel_modg')
#y_spec=librosa.amplitude_to_db(abs(y_modg), np.max)
#plt.figure(figsize=(14,8))
#plt.title("MODG Transform Spectogram \n %s"%files[10])
#librosa.display.specshow(y_spec,sr=fs,y_axis='log', x_axis='time')
#plt.colorbar(format='%+2.0f dB');
#print("Spectogram Array Shape:",y_spec.shape)


def featureGenerator(files, labels):
    while True:
        for i,f in enumerate(files):
            try:
                feature_vectors = []
                label = []
                y, sr = librosa.load(f, sr=fs)
                if len(y) < 2:
                    print("Error loading %s" % f)
                    continue
                y, index = librosa.effects.trim(y,top_db=60) #Trim
                y = normalize(y)
                duration_in_samples=librosa.time_to_samples(1, sr=fs)
                y_pad = librosa.util.fix_length(y, duration_in_samples) #Pad/Trim to same duration
                #y_stft=librosa.core.stft(y_pad, n_fft=n_fft, hop_length=hop_length)
                #y_modg = modg_spectrogram(y_pad, typ = 'modg')
                y_mel = librosa.feature.melspectrogram(y=y_pad, sr=sr, n_fft=N_fft,hop_length=hop_length,
                                      fmax = f_max, n_mels = n_mels)
                y_modg_mel = modg_spectrogram(y_pad, typ = 'mel_power')
                
                y_spec_1=librosa.amplitude_to_db(abs(y_mel), np.max)  #y_stft
                y_spec_2=librosa.amplitude_to_db(abs(y_modg_mel), np.max) #y_modg
                scaler = StandardScaler()
                dtype = K.floatx()
                data_1 = scaler.fit_transform(y_spec_1).astype(dtype)
                data_2 = scaler.fit_transform(y_spec_2).astype(dtype)
                data_1 = np.expand_dims(data_1, axis=0)
                data_2 = np.expand_dims(data_2, axis=0)
                data_1 = np.expand_dims(data_1, axis=3)
                data_2 = np.expand_dims(data_2, axis=3)
                data = np.concatenate((data_1, data_2), 3)
                feature_vectors.append(data)
                label.append([labels[i]])
                yield feature_vectors, label
            except Exception as e:
                print("Error loading %s. Error: %s" % (f,e))
                raise
                break

#nvidia-smi
hist = History();
es = EarlyStopping(monitor='val_acc', min_delta=0.01, restore_best_weights=True, patience= 10, verbose=1 )
mc = ModelCheckpoint('H:\\01_einfaches modell\\1.05_results\\best_modell\\mel_power_best_model.h5', monitor='val_acc',save_best_only=True, verbose=1)

callbacksKeras=[hist,es,mc]
    
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True), log_device_placement=True)
session = tf.Session(config=config)
K.set_session(session)

model.fit_generator(featureGenerator(train_set_files, train_classes), 
                    validation_data=(featureGenerator(test_set_files, test_classes)), 
                    validation_steps=150, 
                    steps_per_epoch=450,epochs=30,callbacks=callbacksKeras, verbose=1)
with open('H:\\01_einfaches modell\\1.05_results\\history\\mel_power_history.pkl', 'wb') as f:
    pickle.dump(hist, f, pickle.HIGHEST_PROTOCOL)
#with open('H:\\02_Einfaches modell_Results\\History\\stft_history.pkl', 'rb') as f:
#    history = pickle.load(f)

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

#path_history = 'H:\\02_Einfaches modell_Results\\History\\power_history.pkl'
#with open(path_history, 'wb') as f:
#        pickle.dump(hist, f, pickle.HIGHEST_PROTOCOL) 

plot_history(hist)

saved_model = load_model('H:\\01_einfaches modell\\1.05_results\\best_modell\\mel_power_best_model.h5')
test_pred = saved_model.predict_generator(featureGenerator(test_set_files, test_classes), steps=150,verbose=1)

predictions_round=np.around(test_pred).astype('int');
predictions_int=np.argmax(predictions_round,axis=1);
predictions_labels=labelencoder.inverse_transform(np.ravel(predictions_int));

# Recall - the ability of the classifier to find all the positive samples
print("Recall: ", recall_score(classes_num[test_index], predictions_int,average=None))

# Precision - The precision is intuitively the ability of the classifier not to 
#label as positive a sample that is negative
print("Precision: ", precision_score(classes_num[test_index], predictions_int,average=None))

# F1-Score - The F1 score can be interpreted as a weighted average of the precision 
#and recall
print("F1-Score: ", f1_score(classes_num[test_index], predictions_int, average=None))

# Accuracy - the number of correctly classified samples
print("Accuracy: %.2f  ," % accuracy_score(classes_num[test_index], predictions_int,normalize=True), accuracy_score(classes_num[test_index], predictions_int,normalize=False) )
print("Number of samples:",classes_num[test_index].shape[0])

print(classification_report(classes_num[test_index], predictions_int))

# Compute confusion matrix
cnf_matrix = confusion_matrix(classes_num[test_index], predictions_int)
np.set_printoptions(precision=2)

# Function to Plot Confusion Matrix
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    """
    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
# Plot non-normalized confusion matrix
plt.figure(figsize=(16,12))
plot_confusion_matrix(cnf_matrix, classes=labelencoder.classes_,
                      title='Confusion matrix, without normalization')

# Find wrong predicted samples indexes
wrong_predictions = [i for i, (e1, e2) in enumerate(zip(classes_num[test_index], predictions_int)) if e1 != e2]

# Find wrong predicted audio files
print(np.array(labels)[test_index[wrong_predictions]])
print(predictions_labels[wrong_predictions].T)
print(np.array(files)[test_index[wrong_predictions]])