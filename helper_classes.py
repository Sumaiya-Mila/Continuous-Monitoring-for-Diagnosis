# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 19:02:56 2021

"""
import vaex
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from csv import reader

#%%
# =============================================================================
#   Class encapsulating functions to save the raw_data (csv) files in appropriate formats. Functions invoked only once during the initial setup. 
# =============================================================================
class SaveFiles:
    
    def __init__(self,
                 path_to_hdf5_folder = None,
                 path_to_raw_data_folder = None,
                 path_to_preprocessed_data_folder = None):
        
        
        if path_to_hdf5_folder is None:
            path_to_hdf5_folder = r"/home/UFAD/mila.s/mila_trident/hdf5_files"
            
        if path_to_raw_data_folder is None:
            path_to_raw_data_folder = r"/home/UFAD/mila.s/mila_trident/Raw Data"
            
        if path_to_preprocessed_data_folder is None:
            path_to_preprocessed_data_folder = r"/home/UFAD/mila.s/mila_trident/preprocessed_files"
        
        self.path_to_hdf5_folder = path_to_hdf5_folder
        self.path_to_preprocessed_data_folder = path_to_preprocessed_data_folder
        self.path_to_raw_data_folder = path_to_raw_data_folder
        
        
        Path(self.path_to_hdf5_folder).mkdir(parents=True, exist_ok=True)
        Path(self.path_to_preprocessed_data_folder).mkdir(parents=True, exist_ok=True)
        
        #self.Convert_to_hdf5()
        self.Preprocess_Measurements()
        #self.Save_Unique_Source_Values()
        #self.Save_Male_Female_IDs()
        self.Save_Covid_Measurements()
        self.Free_Memory()
        return
    
    def Convert_to_hdf5(self):
        
        path_to_csv = self.path_to_raw_data_folder + "/measurement_v6.0.csv"        
        for i, chunk in enumerate(vaex.read_csv(path_to_csv,
                                        chunk_size = 1000000, 
                                        engine = 'python', 
                                        error_bad_lines=False, 
                                        delimiter= '\t')):
            df_chunk = vaex.from_pandas(chunk, copy_index=False)
            export_path = self.path_to_hdf5_folder +  f'/part_{i}.hdf5'
            df_chunk.export_hdf5(export_path)
            print(f"chunk:{i}")
        
        df = vaex.open(self.path_to_hdf5_folder + '/part*')
        df.export_hdf5(self.path_to_hdf5_folder + '/measurements.hdf5')
        print("saved measurements.csv as hdf5 (before pre-processing) ")
        del df
        return
    
    def Preprocess_Measurements(self):
        
        path_to_measurements_hdf5_file = self.path_to_hdf5_folder + '/measurements.hdf5'
        self.measurements_df = vaex.open(path_to_measurements_hdf5_file)
        columns_to_keep = ["MEASUREMENT_SOURCE_VALUE",
                           "MEASUREMENT_TYPE_CONCEPT_ID",
                           "MEASUREMENT_DATE",
                           "MEASUREMENT_DATETIME",
                           "PERSON_ID",
                           "VALUE_SOURCE_VALUE",
                           "UNIT_SOURCE_VALUE",
                           "VALUE_AS_NUMBER","VISIT_OCCURRENCE_ID"]
        self.measurements_df = self.measurements_df[columns_to_keep]
        self.measurements_df.export_hdf5(self.path_to_preprocessed_data_folder + '/measurements.hdf5')
        print("saved pre-processed measurments as hdf5 ")       
                
        return
    
    def Save_Unique_Source_Values(self):
        
        unique_source_values = self.measurements_df.MEASUREMENT_SOURCE_VALUE.to_pandas_series().unique()
        path_to_csv = self.path_to_preprocessed_data_folder + r'\unique_source_values.csv'
        pd.DataFrame(unique_source_values).to_csv(path_to_csv, header=None, index=None)
        del unique_source_values
        
        return

    def Save_Male_Female_IDs(self):

        person_ids_df = pd.read_csv(self.path_to_raw_data_folder + '\person_v6.0.csv', sep='\t')
        person_ids_df = person_ids_df[["PERSON_ID",
                                 "GENDER_SOURCE_VALUE",
                                 "YEAR_OF_BIRTH"]]
        female_ids = person_ids_df['PERSON_ID'].loc[person_ids_df['GENDER_SOURCE_VALUE']=='Female'].values
        male_ids = person_ids_df['PERSON_ID'].loc[person_ids_df['GENDER_SOURCE_VALUE']=='Male'].values
        
        path_to_csv = self.path_to_preprocessed_data_folder + r'\female_ids.csv'
        pd.DataFrame(female_ids).to_csv(path_to_csv,header=None, index=None)
        path_to_csv = self.path_to_preprocessed_data_folder + r'\male_ids.csv'
        pd.DataFrame(male_ids).to_csv(path_to_csv,header=None, index=None)
        
        del male_ids
        del female_ids
        
        return
    
    def Save_Covid_Measurements(self):
#        covid_test_codes = ['94500-6',
#                            '94309-2',
#                            '87635',
#                            'U0001',
#                            'U0002']
        condition_0 = (self.measurements_df.MEASUREMENT_TYPE_CONCEPT_ID == 44818702) 
        condition_1 = (self.measurements_df.MEASUREMENT_SOURCE_VALUE == '94500-6' ) 
        condition_2 = (self.measurements_df.MEASUREMENT_SOURCE_VALUE == '94309-2') 
        condition_3 = (self.measurements_df.MEASUREMENT_SOURCE_VALUE == '87635' ) 
        condition_4 = (self.measurements_df.MEASUREMENT_SOURCE_VALUE == 'U0001' ) 
        condition_5 = (self.measurements_df.MEASUREMENT_SOURCE_VALUE == 'U0002' ) 
        condition_6 = (self.measurements_df.MEASUREMENT_SOURCE_VALUE == '??' ) 
        condition =  condition_0 | condition_1 | condition_2 | condition_3 | condition_4 | condition_5 | condition_6

        self.covid_measurements_df = self.measurements_df[condition].to_pandas_df()
        self.covid_tested_patient_ids = self.covid_measurements_df['PERSON_ID'].unique()
        self.covid_positive_patient_ids = self.covid_measurements_df['PERSON_ID'].loc[self.covid_measurements_df['VALUE_SOURCE_VALUE']=='Detected'].unique()
        self.covid_negative_patient_ids = self.covid_measurements_df['PERSON_ID'].loc[self.covid_measurements_df['VALUE_SOURCE_VALUE']=='Not Detected'].unique()

        self.filtered_measurements_df = self.measurements_df[self.measurements_df.PERSON_ID.isin(self.covid_tested_patient_ids)]

        path_to_csv = self.path_to_preprocessed_data_folder + r'/covid_measurements.csv'
        self.covid_measurements_df.to_csv(path_to_csv,header=None, index=None)
        # path_to_csv = self.path_to_preprocessed_data_folder + r'/covid_positive_patient_ids.csv'
        # pd.DataFrame(self.covid_positive_patient_ids).to_csv(path_to_csv, header=None, index=None)
        # path_to_csv = self.path_to_preprocessed_data_folder + r'/covid_negative_patient_ids.csv'
        # pd.DataFrame(self.covid_negative_patient_ids).to_csv(path_to_csv, header=None, index=None)
        # path_to_csv = self.path_to_preprocessed_data_folder + r'/covid_tested_patient_ids.csv'
        # pd.DataFrame(self.covid_tested_patient_ids).to_csv(path_to_csv, header=None, index=None)
        path_to_hdf5 = self.path_to_preprocessed_data_folder + r'/filtered_measurements.hdf5'
        self.filtered_measurements_df.export_hdf5(path_to_hdf5)

        return
    
    def Free_Memory(self):
        
        del self.measurements_df
        del self.filtered_measurements_df
        del self.covid_measurements_df
        
        return

# =============================================================================
# Class encapsulating data analysis and visulaization functions
# =============================================================================
    
class AnalyzeData:
    
    def __init__(self,
                 path_to_preprocessed_data_folder = None,
                 path_to_raw_data_folder = None):

        if path_to_preprocessed_data_folder is None:
            path_to_preprocessed_data_folder = r"H:\mila_trident\preprocessed_files"

        if path_to_raw_data_folder is None:
            path_to_raw_data_folder = r"H:\mila_trident\Raw Data"
        
                
        
        self.path_to_preprocessed_data_folder = path_to_preprocessed_data_folder
        self.path_to_raw_data_folder = path_to_raw_data_folder
        self.Read_Measurements()
        #self.Read_Person_IDs()
        #self.Read_Male_Female_IDs()
        #self.Read_Covid_Tested_Patient_IDs()
        #self.Save_CovidPatients_Unique_Source_Values()
        self.Get_Covid_Patients_Unique_Source_Value_Count()

        return
    
    
    def Read_Measurements(self):
        self.measurements_df = vaex.open(self.path_to_preprocessed_data_folder + '\\filtered_measurements.hdf5')
        return
    
    def Read_Person_IDs(self):
        self.person_ids_df = pd.read_csv(self.path_to_raw_data_folder + '\person_v6.0.csv', sep='\t')
        self.person_ids_df = self.person_ids_df[["PERSON_ID",
                                 "GENDER_SOURCE_VALUE",
                                 "YEAR_OF_BIRTH"]]
        return
        
    def Read_Male_Female_IDs(self):
        self.female_ids = self.person_ids_df['PERSON_ID'].loc[self.person_ids_df['GENDER_SOURCE_VALUE']=='Female'].values
        self.male_ids = self.person_ids_df['PERSON_ID'].loc[self.person_ids_df['GENDER_SOURCE_VALUE']=='Male'].values
        
        return
        

    def Read_Covid_Tested_Patient_IDs(self):   
        
        self.covid_tested_patient_ids = pd.read_csv(self.path_to_preprocessed_data_folder + r'\covid_tested_patient_ids.csv')
        self.covid_positive_patient_ids = pd.read_csv(self.path_to_preprocessed_data_folder + r'\covid_positive_patient_ids.csv')
        self.covid_negative_patient_ids = pd.read_csv(self.path_to_preprocessed_data_folder + r'\covid_negative_patient_ids.csv')
        
        return 

    def Get_Meaurement_Count(self):
        measurements_count = self.measurements_df.groupby(self.measurements_df.PERSON_ID,
                                                          agg = 'count')
        return measurements_count

    def Save_CovidPatients_Unique_Source_Values(self):
        unique_source_values = self.measurements_df.MEASUREMENT_SOURCE_VALUE.unique()
        path_to_csv = self.path_to_preprocessed_data_folder + r'\covid_patients_unique_source_values.csv'
        pd.DataFrame(unique_source_values).to_csv(path_to_csv, header=None, index=None)
        del unique_source_values

        return

    def Get_Covid_Patients_Unique_Source_Value_Count(self):
        measurement_source_values_name = []
        patient_count = []

        with open(self.path_to_preprocessed_data_folder + r"\covid_patients_unique_source_values.csv", 'r') as read_obj:
            csv_reader = reader(read_obj)

            for row in csv_reader:
                measurement_source_values = row[0]

                selected_measurements_df = self.measurements_df[
                    self.measurements_df.MEASUREMENT_SOURCE_VALUE == measurement_source_values].to_pandas_df()

                measurement_source_values_name.append(row[0])
                patient_count.append(len(selected_measurements_df.PERSON_ID.unique()))

        d = {'MEASUREMENT_SOURCE_VALUE': measurement_source_values_name, 'Count': patient_count}
        selected_measurements_count_df = pd.DataFrame(d)
        # selected_measurements_count_df.to_csv(path_to_preprocessed_data_folder + '\measurement_count_on_unique_patients.csv',index=None)

    def Get_Selected_Measurements(self,
                                 measurement_source_value):
        selected_measurements_df = self.measurements_df[self.measurements_df.MEASUREMENT_SOURCE_VALUE == measurement_source_value]

        selected_measurements_count = selected_measurements_df.groupby(self.measurements_df.PERSON_ID,
                                                          agg='count')
        return selected_measurements_count

        
    
    
    
    
    
    
    
    
    
    
    
    
    
