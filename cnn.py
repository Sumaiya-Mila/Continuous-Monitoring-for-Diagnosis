#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 17:34:34 2022

@author: mila.s
"""

import pandas as pd
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
from os import listdir
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from tensorflow.keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

path_to_Visit_Occurrence_folder = r"/home/UFAD/mila.s/mila_trident/preprocessed_files/Files_with_Visit_Occurrence"

############retrieving train set, padding and slicing

path = path_to_Visit_Occurrence_folder+ '/Train_dropped'
train = list()
for i in range(1,3489):
    file_path = path + '/patient_data' + str(i) + '.csv'
    df = pd.read_csv(file_path, header=0)
    # df = df.iloc[:,1:-1]
    df = df.iloc[:,[1,3,4,5]]
    values = df.values
    train.append(values)

# padding to train dropped data 17267, 8634,4317,2878 - 1h freq, all feat 2h freq

to_pad = 17267
new_seq = []
for one_seq in train:
    len_one_seq = len(one_seq)
    last_val = one_seq[-1]
    n = to_pad - len_one_seq
   
    to_concat = np.repeat(one_seq[-1], n).reshape(4, n).transpose()
    new_one_seq = np.concatenate([one_seq, to_concat])
    new_seq.append(new_one_seq)
final_train = np.stack(new_seq)

# truncate the train dropped data 860, 430,215,144 
seq_len = 860
final_train=sequence.pad_sequences(final_train, maxlen=seq_len, padding='post', dtype='float', truncating='post')
final_train = np.array(final_train)
print(final_train.shape)


train_target = pd.read_csv(path+'/train_data_target.csv', header=0)
train_target = train_target.values[:,1]
train_target = np.array(train_target)
# print(type(train_target))
# print(train[0])

############retrieving test set, padding and slicing
path = path_to_Visit_Occurrence_folder+ '/Test_dropped'
test = list()
for i in range(1,1092):
    file_path = path + '/patient_data' + str(i) + '.csv'
    df = pd.read_csv(file_path, header=0)
    # df = df.iloc[:,1:-1]
    df = df.iloc[:,[1,3,4,5]]
    values = df.values
    test.append(values)
    
# padding to test dropped data 8405, 4202 ,2101,1400
to_pad = 8405
new_seq = []
for one_seq in test:
    len_one_seq = len(one_seq)
    last_val = one_seq[-1]
    n = to_pad - len_one_seq
   
    to_concat = np.repeat(one_seq[-1], n).reshape(4, n).transpose()
    new_one_seq = np.concatenate([one_seq, to_concat])
    new_seq.append(new_one_seq)
final_test = np.stack(new_seq)

# truncate the dropped data 860, 430,215,144
seq_len = 860
final_test=sequence.pad_sequences(final_test, maxlen=seq_len, padding='post', dtype='float', truncating='post')
final_test = np.array(final_test)
# print(final_test)

test_target = pd.read_csv(path+'/test_data_target.csv', header=0)
test_target = test_target.values[:,1]
test_target = np.array(test_target)
# print(type(test_target))


############retrieving validation set, padding and slicing    
path = path_to_Visit_Occurrence_folder+ '/Validation_dropped'
val = list()
for i in range(1,873):
    file_path = path + '/patient_data' + str(i) + '.csv'
    df = pd.read_csv(file_path, header=0)
    # df = df.iloc[:,1:-1]
    df = df.iloc[:,[1,3,4,5]]
    values = df.values
    val.append(values)

# padding to val dropped data 10040, 5020,2510,1673
to_pad = 10040
new_seq = []
for one_seq in val:
    len_one_seq = len(one_seq)
    last_val = one_seq[-1]
    n = to_pad - len_one_seq
   
    to_concat = np.repeat(one_seq[-1], n).reshape(4, n).transpose()
    new_one_seq = np.concatenate([one_seq, to_concat])
    new_seq.append(new_one_seq)
final_val = np.stack(new_seq)

# truncate the dropped data 860, 430,215,144
seq_len = 860
final_val=sequence.pad_sequences(final_val, maxlen=seq_len, padding='post', dtype='float', truncating='post')
final_val = np.array(final_val)

val_target = pd.read_csv(path+'/val_data_target.csv', header=0)
val_target = val_target.values[:,1]
val_target = np.array(val_target)

# print(val_target)
# print(final_test)
# print(test_target)


 
#######timeseries model train using cnn
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


def create_model(optimizer='adam', init='glorot_uniform'):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(seq_len, 4)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create Grid Search
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)

# Fit the model
grid_result = grid.fit(final_train, train_target)

# Summarize results
print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(f"Mean: {mean:.2f} Std Dev: ({stdev:.2f}) with: {param}")

# Load the best model
best_model = grid_result.best_estimator_.model
test_preds = best_model.predict(final_test)
test_preds = np.round(test_preds).astype(int)
# print(test_preds)
import numpy as np
from sklearn.metrics import accuracy_score,f1_score, recall_score, precision_score
print(accuracy_score(test_target, test_preds))
print(f1_score(test_target, test_preds))
print(precision_score(test_target, test_preds))
print(recall_score(test_target, test_preds))
##################model train using encoder-decoder
# from keras import layers
# from keras.layers import TimeDistributed

# enco_deco = Sequential()
# # Encoder
# enco_deco.add(LSTM(32, input_shape=(seq_len, 117),return_sequences=True))
# enco_deco.add(LSTM(units=16,return_sequences=True))
# enco_deco.add(LSTM(units=8))

# #feature vector
# enco_deco.add(layers.RepeatVector(32))

# enco_deco.add(LSTM(units=32,return_sequences=True))
# enco_deco.add(LSTM(units=16,return_sequences=True))
# enco_deco.add(TimeDistributed(tf.keras.layers.Dense(1, activation='sigmoid')))
# enco_deco.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
# enco_deco.summary()
# history = enco_deco.fit(final_train,train_target, epochs=25,steps_per_epoch=20, validation_data=(final_val,val_target), verbose =1)
# test_preds = enco_deco.predict(final_test)
# test_preds = np.round(test_preds).astype(int)
# print(test_preds)
# print(test_preds.shape)

# l = list()
# for i in range(0,test_preds.shape[0]):
#     vals,counts = np.unique(test_preds[i], return_counts=True)
#     index = np.argmax(counts)
#     # print(vals[index])
#     l.append(vals[index])
# print(np.count_nonzero(test_preds == 1))
# import numpy as np
# from sklearn.metrics import accuracy_score,f1_score, recall_score, precision_score
# print(accuracy_score(test_target, l))
# print(f1_score(test_target, l))
# print(precision_score(test_target, l))
# print(recall_score(test_target, l))



