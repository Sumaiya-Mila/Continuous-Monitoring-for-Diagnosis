#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 16:12:03 2021

@author: mila.s
"""
import vaex
import pandas as pd
from sklearn.model_selection import train_test_split


class ProcessDaTa:
    def __init__(self,path_to_preprocessed_data_folder=None,
                 path_to_Train_Test_data_folder=None,
                 path_to_Undersample_data_folder=None):

        if path_to_preprocessed_data_folder is None:
            path_to_preprocessed_data_folder = r"/home/UFAD/mila.s/mila_trident/preprocessed_files"

        if path_to_Train_Test_data_folder is None:
            path_to_Train_Test_data_folder = r"/home/UFAD/mila.s/mila_trident/preprocessed_files/Train_Test_Data"
            
        if path_to_Undersample_data_folder is None:
            path_to_Undersample_data_folder = r"/home/UFAD/mila.s/mila_trident/preprocessed_files/Undersample_Data"

        self.path_to_preprocessed_data_folder = path_to_preprocessed_data_folder
        self.path_to_Train_Test_data_folder = path_to_Train_Test_data_folder
        self.path_to_Undersample_data_folder = path_to_Undersample_data_folder
        
        self.Store_Monitorable_Features()
        # self.Split_Train_Test_Data()
        
        return
    
    def Split_Train_Test_Data(self):
        ##########################################################
        ###splitted train test data of unique and valued patients
        ##########################################################
        
        Xdata = pd.read_csv(self.path_to_preprocessed_data_folder +'/patient_vs_feature_value_merged.csv')
        patient_id = pd.read_csv(self.path_to_Undersample_data_folder + '/underSampled_patient_id.csv')      
        patient_id = patient_id.iloc[:,-1]
        
        new = Xdata["PERSON_ID"].isin(list(patient_id))
        Xdata = Xdata[new]
        ydata = Xdata.iloc[:, -2:-1]
        
        X_train, X_test, y_train, y_test = train_test_split(Xdata, ydata ,
                                                                     test_size= 0.2, random_state=42)
                
        # X_train.to_csv(self.path_to_Train_Test_data_folder+ '/X_train_final.csv', index=None)
        # X_test.to_csv(self.path_to_Train_Test_data_folder+ '/X_test_final.csv', index=None)
        # y_train.to_csv(self.path_to_Train_Test_data_folder+ '/y_train_final.csv', index=None)
        # y_test.to_csv(self.path_to_Train_Test_data_folder+ '/y_test_final.csv', index=None)
        return
    
    
    def Store_Monitorable_Features(self):
        ######################################################################
        ####Seperated the patients with no feature value, stored patient ids, 
        ####stored all the feature values of the patients in hdf5 file
        ######################################################################
        
        measurements_df = vaex.open(self.path_to_preprocessed_data_folder + '/filtered_measurements.hdf5')
        measurements_df = measurements_df["MEASUREMENT_SOURCE_VALUE","PERSON_ID","VALUE_AS_NUMBER","VALUE_SOURCE_VALUE"]
        m_s_v1 = measurements_df.MEASUREMENT_SOURCE_VALUE == 'TEMPERATURE'
        m_s_v2 = measurements_df.MEASUREMENT_SOURCE_VALUE == 'HEART RATE'
        m_s_v3 = measurements_df.MEASUREMENT_SOURCE_VALUE == 'MAP - Cuff'
        m_s_v4 = measurements_df.MEASUREMENT_SOURCE_VALUE == 'RESP RATE'
        m_s_v5 = measurements_df.MEASUREMENT_SOURCE_VALUE == 'ETCO2'
        m_s_v6 = measurements_df.MEASUREMENT_SOURCE_VALUE == 'SPO2'
        
        m_s_v = m_s_v1|m_s_v2|m_s_v3|m_s_v4|m_s_v5|m_s_v6
        
        measurements_df = measurements_df[m_s_v]        
        
        X_train = pd.read_csv(self.path_to_Train_Test_data_folder+ '/X_train_Under_Sample.csv')
        X_test = pd.read_csv(self.path_to_Train_Test_data_folder+ '/X_test_Under_Sample.csv')
        p_train = X_train.iloc[:, -1]
        p_test = X_test.iloc[:, -1]
        patient_list = (p_train.append(p_test)) #list of patient id of my interest
        
        measurements_df = measurements_df[measurements_df.PERSON_ID.isin(patient_list)]
        patient_id = pd.DataFrame(measurements_df.PERSON_ID.unique())
        print(patient_id)
        print(len(patient_id))
        # patient_id.to_csv(self.path_to_Undersample_data_folder + '/underSampled_patient_id.csv', index=None)
        
        measurements_df.export_hdf5(self.path_to_Undersample_data_folder+
                                    "/patient_monitorable_feature_value.hdf5")
        
        return
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    