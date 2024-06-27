#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 17:30:02 2022

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
import random

path_to_Visit_Occurrence_folder = r"/home/UFAD/mila.s/mila_trident/preprocessed_files/Files_with_Visit_Occurrence"

############retrieving train set, padding and slicing

path = path_to_Visit_Occurrence_folder+ '/Train_dropped'
train = list()
for i in range(1,3489):
    file_path = path + '/patient_data' + str(i) + '.csv'
    df = pd.read_csv(file_path, header=0)
    # df = df.iloc[:,1:-1]
    
    # df = df.iloc[:,[1,3,4,5]]
    # hr = (df.iloc[:,4])/5
    # hr=hr.rename('resp')
    # # print((hr))
    # df = df.iloc[:,[1,2,4,5]]
    # df = df.join(hr)
    
    #incorporating noise to bp
    bp = df.iloc[:,5]   #storing original bp value from dataset
    noise_df = pd.DataFrame([random.uniform(-1,6.92) 
                             for m in range(len(bp))], columns=['random noise'])    #converting list of noise(randomly generated within range) to dataframe
    #adding the original and noise and storing in another df
    bp_w_noise = (bp.iloc[:] + noise_df.iloc[:,0]).to_frame() #adding random noise to each bp value and storing to another dataframe
    bp_w_noise = bp_w_noise.rename(columns= {0: 'MAP'}) #renaming column of the df
    df = df.iloc[:,[1,2,3,4]] #df stores 4 features(wo bp)
    df = df.join(bp_w_noise) #df appends bp with noise as a column
    # df = df.join(hr)
    
    values = df.values
    train.append(values)
   
# print(train) 
# padding to train dropped data 17267, 8634,4317,2878 - 1h freq, all feat 2h freq

to_pad = 17267
new_seq = []
for one_seq in train:
    len_one_seq = len(one_seq)
    last_val = one_seq[-1]
    n = to_pad - len_one_seq
   
    to_concat = np.repeat(one_seq[-1], n).reshape(5, n).transpose()
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
    # df = df.iloc[:,[1,3,4,5]]
    
    # hr = (df.iloc[:,4])/5
    # hr=hr.rename('resp')
    # # print((hr))
    # df = df.iloc[:,[1,2,4,5]]
    # df = df.join(hr)
    
    bp = df.iloc[:,5]   #storing original bp value from dataset
    noise_df = pd.DataFrame([random.uniform(-1,6.92) 
                             for m in range(len(bp))], columns=['random noise'])    #converting list of noise(randomly generated within range) to dataframe
    #adding the original and noise and storing in another df
    bp_w_noise = (bp.iloc[:] + noise_df.iloc[:,0]).to_frame() #adding random noise to each bp value and storing to another dataframe
    bp_w_noise = bp_w_noise.rename(columns= {0: 'MAP'}) #renaming column of the df
    df = df.iloc[:,[1,2,3,4]] #df stores 4 features(wo bp)
    df = df.join(bp_w_noise) #df appends bp with noise as a column
    # df = df.join(hr)
    
    values = df.values
    test.append(values)
    
# padding to test dropped data 8405, 4202 ,2101,1400
to_pad = 8405
new_seq = []
for one_seq in test:
    len_one_seq = len(one_seq)
    last_val = one_seq[-1]
    n = to_pad - len_one_seq
   
    to_concat = np.repeat(one_seq[-1], n).reshape(5, n).transpose()
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
    # df = df.iloc[:,[1,3,4,5]]
    
    # hr = (df.iloc[:,4])/5
    # hr=hr.rename('resp')
    # # print((hr))
    # df = df.iloc[:,[1,2,4,5]]
    # df = df.join(hr)
    
    bp = df.iloc[:,5]   #storing original bp value from dataset
    noise_df = pd.DataFrame([random.uniform(-1,6.92) 
                             for m in range(len(bp))], columns=['random noise'])    #converting list of noise(randomly generated within range) to dataframe
    #adding the original and noise and storing in another df
    bp_w_noise = (bp.iloc[:] + noise_df.iloc[:,0]).to_frame() #adding random noise to each bp value and storing to another dataframe
    bp_w_noise = bp_w_noise.rename(columns= {0: 'MAP'}) #renaming column of the df
    df = df.iloc[:,[1,2,3,4]] #df stores 4 features(wo bp)
    df = df.join(bp_w_noise) #df appends bp with noise as a column
    # df = df.join(hr)
    
    values = df.values
    val.append(values)

# padding to val dropped data 10040, 5020,2510,1673
to_pad = 10040
new_seq = []
for one_seq in val:
    len_one_seq = len(one_seq)
    last_val = one_seq[-1]
    n = to_pad - len_one_seq
   
    to_concat = np.repeat(one_seq[-1], n).reshape(5, n).transpose()
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

final_train = np.append(final_train, final_val,axis = 0)
train_target = np.append(train_target, val_target)
print(train_target.shape)
# print(final_val)

from sktime.classification.all import *
from sklearn.metrics import accuracy_score,f1_score, recall_score, precision_score
import pickle
from sktime.classification.interval_based import CanonicalIntervalForest,DrCIF
import time

ftr = np.transpose(final_train,(0,2,1))
fts = np.transpose(final_test,(0,2,1))
# print((final_train))
print(ftr.shape)
print(fts.shape)

###################SupervisedTimeSeriesForest
import time
import pickle
from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.classification.interval_based import SupervisedTimeSeriesForest
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def create_model(n_estimators):
    a = []
    for i in range(0, 5):
        a.append((str(i), SupervisedTimeSeriesForest(n_estimators=n_estimators), [i]))
    clf = ColumnEnsembleClassifier(estimators=a)
    return clf

begin = time.time()

# Create a model instance
model = create_model(n_estimators=30)

param_grid = {
    'n_estimators': [10, 20, 30, 50]  
}

# Wrap the model with GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid={'estimator__n_estimators': param_grid['n_estimators']}, cv=3, n_jobs=-1, verbose=1)

grid_result = grid_search.fit(ftr, train_target)

print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(f"Mean: {mean:.2f} Std Dev: ({stdev:.2f}) with: {param}")

best_model = grid_result.best_estimator_

# Save the best model
filename = 'SupervisedTimeSeriesForest_dropped_w_bpnoise4.sav'
pickle.dump(best_model, open(filename, 'wb'))
print('model dumped')

# Load the best model
loaded_model = pickle.load(open(filename, 'rb'))
print('model loaded')

# Predict with the best model
pred = loaded_model.predict(fts)

# Evaluate the predictions
print(f"Accuracy: {accuracy_score(test_target, pred)}")
print(f"F1 Score: {f1_score(test_target, pred)}")
print(f"Precision: {precision_score(test_target, pred)}")
print(f"Recall: {recall_score(test_target, pred)}")

end = time.time()
print('time taken - ')
print(end - begin)

###################TimeSeriesForestClassifier
def create_tsf_model(n_estimators):
    a = []
    for i in range(0, 5):
        a.append((str(i), TimeSeriesForestClassifier(n_estimators=n_estimators), [i]))
    clf = ColumnEnsembleClassifier(estimators=a)
    return clf

begin = time.time()

model = create_tsf_model(n_estimators=25)

# Define the grid search parameters
param_grid = {
    'estimator__n_estimators': [10, 25, 50, 100] 
}

# Wrap the model with GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)

# Fit the model using GridSearchCV
grid_result = grid_search.fit(ftr, train_target)

# Summarize the results
print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(f"Mean: {mean:.2f} Std Dev: ({stdev:.2f}) with: {param}")

# Get the best model from grid search
best_model = grid_result.best_estimator_

# Save the best model
filename = 'TimeSeriesForestClf_dropped_wo_resp_h3.sav'
pickle.dump(best_model, open(filename, 'wb'))
print('model dumped')

# Load the best model
loaded_model = pickle.load(open(filename, 'rb'))
print('model loaded')

# Predict with the best model
pred = loaded_model.predict(fts)

# Evaluate the predictions
print(f"Accuracy: {accuracy_score(test_target, pred)}")
print(f"F1 Score: {f1_score(test_target, pred)}")
print(f"Precision: {precision_score(test_target, pred)}")
print(f"Recall: {recall_score(test_target, pred)}")

end = time.time()
print('time taken - ')
print(end - begin)
###################IndividualBOSS

begin = time.time()
a = []
for i in range (0,117):
    a.append((str(i),IndividualBOSS(window_size=2),[i]))
a = tuple(a)
a= list(a)   
clf=ColumnEnsembleClassifier(estimators=a)
print('clf done')

# clf = ColumnEnsembleClassifier(estimators=[
#     ("BOSS0", IndividualBOSS(window_size=10), [0]),
#     # ("BOSS1", IndividualBOSS(window_size=5), [1]),
#     ("BOSS2", IndividualBOSS(window_size=10), [2]),
#     ("BOSS3", IndividualBOSS(window_size=10), [3]),
#     ("BOSS4", IndividualBOSS(window_size=10), [4])
# ])
clf.fit(ftr, train_target)
print('model fit done')

filename = 'BOSS_dropped_all_h4.sav'
pickle.dump(clf, open(filename, 'wb'))
print('model dumped')
loaded_model = pickle.load(open(filename, 'rb'))
print('model loaded')


pred = clf.predict(fts)
print(accuracy_score(pred,test_target))
print(f1_score(pred,test_target))
print(precision_score(pred,test_target))
print(recall_score(pred,test_target))
end = time.time()
print('time taken - ')
print(end-begin)


###################KNeighborsTimeSeriesClassifier
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
# ftr = np.transpose(final_train,(0,2,1))
# fts = np.transpose(final_test,(0,2,1))
# fts = fts.astype(np.float64)
# ftr = ftr.astype(np.float64)
# fts = fts.astype('float32')
# fts2 = np.trunc(fts)
# ftr2 = np.trunc(ftr)
# print(fts2[0,0,0])
# print(str(type(fts2[0,0,0])))
# ftr = np.transpose(ftr,(0,2,1))
# fts = np.transpose(fts,(0,2,1))
print(ftr.shape)
begin = time.time()
clf = clf = ColumnEnsembleClassifier(estimators=[
    ("BOSS", RandomIntervalSpectralEnsemble(n_estimators=10), [3]),
])
clf.fit(ftr, train_target)

filename = 'KNN_dropped_all_h.sav'
pickle.dump(clf, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))
loaded_model2 = loaded_model

pred = loaded_model2.predict(fts)
print(accuracy_score(pred,test_target))
print(f1_score(pred,test_target))
print(precision_score(pred,test_target))
print(recall_score(pred,test_target))
end = time.time()
print('time taken - ')
print(end-begin)

##########################################################
###incorporating random error to bp-map
##########################################################
import random
data = [10,20,30,40,50,60]
  
# Create the pandas DataFrame with column name is provided explicitly
df = pd.DataFrame(data, columns=['Numbers'])
  
# print dataframe.
print(df.iloc[:,0])

#creating a list of random numbers between two specified number
rn = [random.uniform(-1,6.92) for m in range(len(df))]

#creating a df with the random error list. cause without the df, summation operation was giving null value
df2 = pd.DataFrame(rn, columns=['random noise'])
print(df2)

#adding the original and noise and storing in another df
df1 = df.iloc[:,0] + df2.iloc[:,0] 
print(df1)