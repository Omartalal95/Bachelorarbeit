# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 12:02:41 2019

@author: so3524
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 15:31:11 2019

@author: omar elaiashy
"""
import numpy as np
import os
from six.moves import cPickle as pickle #for performance
import matplotlib.pyplot as plt

thres_dict = {0:0.5, 1:0.5, 2:0.5, 3:0.5, 4:0.5, 5:0.5, 6:0.5}

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

def predictions_round(pred_labels, thres_dic):
    
    for i in np.arange(pred_labels.shape[0]):
        for m in np.arange(pred_labels.shape[2]):
            for n in np.arange(pred_labels.shape[3]):
            
                if pred_labels[i][0][m][0] >= thres_dic[0]:
                    pred_labels[i][0][m][0] = 1
                else:
                    pred_labels[i][0][m][0] = 0
            
                if pred_labels[i][0][m][1] >= thres_dic[1]:
                    pred_labels[i][0][m][1] = 1
                else:
                    pred_labels[i][0][m][1] = 0
            
                if pred_labels[i][0][m][2] >= thres_dic[2]:
                    pred_labels[i][0][m][2] = 1
                else:
                    pred_labels[i][0][m][2] = 0
        
                if pred_labels[i][0][m][3] >= thres_dic[3]:
                    pred_labels[i][0][m][3] = 1
                else:
                    pred_labels[i][0][m][3] = 0    
                
                if pred_labels[i][0][m][4] >= thres_dic[4]:
                    pred_labels[i][0][m][4] = 1
                else:
                    pred_labels[i][0][m][4] = 0
                
                if pred_labels[i][0][m][5] >= thres_dic[5]:
                    pred_labels[i][0][m][5] = 1
                else:
                    pred_labels[i][0][m][5] = 0
            
                if pred_labels[i][0][m][6] >= thres_dic[6]:
                    pred_labels[i][0][m][6] = 1
                else:
                    pred_labels[i][0][m][6] = 0
    
    return pred_labels

def binary_conf_matrix(label_actual, label_predicted):
    
    conf_matrix_dict = dict()
    
    for m in np.arange(label_actual.shape[3]):
        
        conf_matrix = np.zeros(shape=(2,2))
          
        for n in np.arange(label_actual.shape[0]): 
  
            for i in np.arange(label_actual.shape[2]):
    
                if label_actual[n][0][i][m] == label_predicted[n][0][i][m]:

                    if label_actual[n][0][i][m]:
                        conf_matrix[0][0] += 1
                    else:
                        conf_matrix[1][1] += 1
    
                else:
        
                    if label_actual[n][0][i][m]:
                        conf_matrix[0][1] += 1
                    else:
                        conf_matrix[1][0] += 1
                        
        conf_matrix_dict[m] = conf_matrix
    
    return conf_matrix_dict

def metrics_eval(label_actual, label_predicted, save_name):
    
    conf_matrix_dict = binary_conf_matrix(label_actual, label_predicted) 
    
    n_inst = len(conf_matrix_dict)
    
    metric_dict = dict()
    
    for i in np.arange(n_inst):
    
        metric_dict_inst = dict()
        conf_matrix = conf_matrix_dict[i]
        
        if (conf_matrix[0][0] + conf_matrix[1][0]) == 0 or (conf_matrix[0][0] + conf_matrix[0][1]) == 0:
            precision = 100
            recall = 100
            f1_score = 100
        else:
            precision = conf_matrix[0][0]/(conf_matrix[0][0] + conf_matrix[1][0])
            recall    = conf_matrix[0][0]/(conf_matrix[0][0] + conf_matrix[0][1])
            f1_score  = 2 * (precision*recall)/(precision + recall)
        
        metric_dict_inst['precision'] = precision
        metric_dict_inst['recall']    = recall
        metric_dict_inst['f1_score']  = f1_score
        
        metric_dict[i] = metric_dict_inst 
    
    name_metric_dict = save_name + '_metric_dict.pkl'
    name_conf_matrix_dict = save_name + '_conf_matrix_dict.pkl'
    
    save_path_metric_dict = os.path.join('D:\\Omar_Elaiashy\\07_Evaluation_Musicnet', name_metric_dict) 
    save_path_conf_matrix_dict = os.path.join('D:\\Omar_Elaiashy\\07_Evaluation_Musicnet', name_conf_matrix_dict)
    
    with open(save_path_metric_dict, 'wb') as f:
        pickle.dump(metric_dict, f)
    with open(save_path_conf_matrix_dict, 'wb') as f:
        pickle.dump(conf_matrix_dict, f)    
    
    return  metric_dict

def plot_predicted_insts(pred_label, n_time_intervals, save_name, edge_linewidths = 0.0):

    dm = pred_label.shape[0]*pred_label.shape[2] # 59*30    
    time_predicted_insts = np.ndarray((7, dm), dtype = int)     

    for i in np.arange(pred_label.shape[0]):
        time_predicted_insts[:,i*pred_label.shape[2]:(i+1)*pred_label.shape[2]] = (pred_label[i][0]).T
    
    fig, ax = plt.subplots(figsize=(60,5))
    
    plt.yticks(np.arange(8), ('Piano', 'Violin', 'Viola', 'Cello', 'Horn', 'Bassoon', 'Clarient'))
    
    x = np.arange(0,((pred_label.shape[0]*3)+100e-3),100e-3)
    step_x = int(len(x) / (n_time_intervals - 1))
    x_positions = np.arange(0,len(x),step_x) 
    x_labels = x[::step_x]
    plt.xticks(x_positions, x_labels)
    
    c = ax.pcolormesh(time_predicted_insts, cmap=plt.cm.binary, edgecolors='r', linewidths=edge_linewidths)
    
    
    plt.gca().invert_yaxis()
    fig.tight_layout()
    
    #ax.set_xlabel('time', fontsize=10)
    #ax.set_ylabel('Predicted Instrument', fontsize=10)
    #ax.set_title('Prediction', fontsize=15)    
    
    name = save_name + '.png'
    save_path = os.path.join('D:\\Omar_Elaiashy\\07_Evaluation_Musicnet', name) 
    plt.savefig(save_path)
    
    


with open('D:\\Omar_Elaiashy\\03_musicnet_modell\\data\\testset\\2556.pkl', 'rb') as f: #todo
    audio_dict_song = pickle.load(f)

to_predict_batch_song = audio_dict_song['mel+mel-modg'] #todo
n_intervals = audio_dict_song['batch_size'] + 1

#saved_model = load_model('D:\\Omar_Elaiashy\\04_musicnet_Results\\modell\\model_mel_mel-modg.hdf5') #todo
saved_model = load_model('D:\\Omar_Elaiashy\\04_musicnet_Results\\best_modell\\best_model_mel_mel-modg.hdf5') #todo

test_pred_song = saved_model.predict(x = to_predict_batch_song, batch_size = 1)

label_actual = audio_dict_song['label'] 
label_predicted = predictions_round(test_pred_song, thres_dict)

metric_dict = metrics_eval(label_actual, label_predicted, save_name = '2556_mel+mel-modg') #todo

plot_predicted_insts(label_predicted, n_time_intervals = n_intervals, save_name = '2556_mel+mel-modg', edge_linewidths = 0.0) #todo

conf = binary_conf_matrix(label_actual, label_predicted)

plot_predicted_insts(np.abs(label_predicted-label_actual), n_time_intervals = n_intervals, 
                     save_name = '2556_false_mel+mel-modg', edge_linewidths = 0.0)#todo
#predictions_round[45]
#test_pred[40]
#with open('D:\\Omar_Elaiashy\\03_musicnet_modell\\data\\testset\\2628.pkl', 'rb') as f:
#    audio_dict_compare = pickle.load(f)
#stft_modg_label = audio_dict_compare['label'] 
#predictions_round[6] - stft_modg_label[6]
##test_pred[40]
#data = np.load('D:\\Omar_Elaiashy\\yte.npy')