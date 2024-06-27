#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 18:57:04 2022

@author: mila.s
"""

import pandas as pd
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
from os import listdir
import tensorflow as tf

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from tensorflow.keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

path_to_Visit_Occurrence_folder = r"/home/UFAD/mila.s/mila_trident/preprocessed_files/Files_with_Visit_Occurrence"

############retrieving train set, padding and slicing

path = path_to_Visit_Occurrence_folder+ '/Train_dropped_all'
train = list()
for i in range(1,3489):
    file_path = path + '/patient_data' + str(i) + '.csv'
    df = pd.read_csv(file_path, header=0)
    df = df.iloc[:,[1,3,4,5]]
    values = df.values
    train.append(values)

# padding to train data 15310,7655, 8634 - 1h freq, 2hfreq, all feat 2h freq
# padding to train dropped data 15310, 7655 - 1h freq, all feat 2h freq

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

#truncate the sequence to length 374, 187, 491 - 1h freq, 2hfreq, all feat 2h freq
# truncate the train dropped data 860, 430 - 1h freq, all feat 2h freq
seq_len = 860
final_train=sequence.pad_sequences(final_train, maxlen=seq_len, padding='post', dtype='float', truncating='post')
final_train = np.array(final_train)
# print(final_train.shape)


train_target = pd.read_csv(path+'/train_data_target.csv', header=0)
train_target = train_target.values[:,1]
train_target = np.array(train_target)
# print(type(train_target))
# print(train[0])

############retrieving test set, padding and slicing
path = path_to_Visit_Occurrence_folder+ '/Test_dropped_all_6h'
test = list()
for i in range(1,1092):
    file_path = path + '/patient_data' + str(i) + '.csv'
    df = pd.read_csv(file_path, header=0)
    df = df.iloc[:,[1,3,4,5]]
    values = df.values
    test.append(values)
    
# padding to test data 10020, 5010, 5020 - 1h freq, 2hfreq, all feat 2h freq
# padding to test dropped data 8405, 4202 - 1h freq, all feat 2h freq
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

#truncate the sequence to length 374, 187, 491- 1h freq, 2hfreq, all feat 2h freq
# truncate the dropped data 860, 430 - 1h freq, all feat 2h freq
seq_len = 860
final_test=sequence.pad_sequences(final_test, maxlen=seq_len, padding='post', dtype='float', truncating='post')
final_test = np.array(final_test)
# print(final_test)

test_target = pd.read_csv(path+'/test_data_target.csv', header=0)
test_target = test_target.values[:,1]
test_target = np.array(test_target)
# print(type(test_target))


############retrieving validation set, padding and slicing    
path = path_to_Visit_Occurrence_folder+ '/Validation_dropped_all'
val = list()
for i in range(1,873):
    file_path = path + '/patient_data' + str(i) + '.csv'
    df = pd.read_csv(file_path, header=0)
    # df = df.iloc[:,1:-1]
    df = df.iloc[:,[1,3,4,5]]
    values = df.values
    val.append(values)

# padding to val data 8405, 4202, 4202- 1h freq, 2hfreq, all feat 2h freq
# padding to val dropped data 10020, 5010 - 1h freq, all feat 2h freq
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

#truncate the sequence to length 374, 187, 491- 1h freq, 2hfreq, all feat 2h freq
# truncate the dropped data 860, 430 - 1h freq, all feat 2h freq
seq_len = 860
final_val=sequence.pad_sequences(final_val, maxlen=seq_len, padding='post', dtype='float', truncating='post')
final_val = np.array(final_val)
print(final_train.shape)

val_target = pd.read_csv(path+'/val_data_target.csv', header=0)
val_target = val_target.values[:,1]
val_target = np.array(val_target)
# print(val_target)
# print(final_test)
# print(test_target)

#######timeseries model train
model = Sequential()
model.add(LSTM(64, input_shape=(seq_len, 4)))#prev 256
# model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

adam = Adam(lr=0.01) #previously 0.001
metric = 'val_accuracy'
chk = ModelCheckpoint('best_model_dropped_wo_spo2_2h7.pkl', monitor=metric, save_best_only=True, mode='max', verbose=1)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
model.fit(final_train, train_target, epochs=20, batch_size=128, callbacks=[chk], validation_data=(final_val,val_target))  #epoch=200

#loading the model and checking accuracy on the test data
model = load_model('best_model_dropped_wo_spo2_2h7.pkl')

from sklearn.metrics import accuracy_score,f1_score, recall_score, precision_score
test_preds = model.predict(final_test)
test_preds = np.round(test_preds).astype(int)
print(accuracy_score(test_target, test_preds))
print(f1_score(test_target, test_preds))
print(precision_score(test_target, test_preds))
print(recall_score(test_target, test_preds))

len_sequences = []
count=1
for one_seq in test:
    len_sequences.append(len(one_seq))
    if len(one_seq)==0:
        print(count)
        # print(one_seq)
    count=count+1
        
print(pd.Series(len_sequences).describe())

print(type(final_test))
print(np.isfinite(final_test).all())
