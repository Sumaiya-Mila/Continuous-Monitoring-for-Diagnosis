# -*- coding: utf-8 -*-
"""
@author: mila.s
"""

import pandas as pd
import numpy as np
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

path_to_Visit_Occurrence_folder = r"C:/Users/mila.s/Downloads/Research/trident/codes and all/preprocessed_files/Files_with_Visit_Occurrence"

############storing the train,test and validation set patient's ids 

path = path_to_Visit_Occurrence_folder+ '/Validation_dropped_6h'
train_id = list()
for i in range(1,873):
    file_path = path + '/patient_data' + str(i) + '.csv'
    df = pd.read_csv(file_path, header=0)
    train_id.append(df.PERSON_ID[0])
    
    continue
   
#Convert the list to a DataFrame
df_train_id = pd.DataFrame(train_id, columns=['PERSON_ID'])

#Specify the filename for the output CSV
output_filename = 'validation_id.csv'

#Write the DataFrame to a CSV file
df_train_id.to_csv(output_filename, index=False)


#########################################################################
#############    KEEPING THE PERSONS OF INTERESTS FROM THE PERSON DATASET
#########################################################################
path_to_person_ds = r"C:/Users/mila.s/Downloads/Research/trident/codes and all/Raw Data/COVID_OMOP_dataset_v6.0/clinical_data_tables_v6.0"

person_df = pd.read_excel(path_to_person_ds+'/person.xlsx')
print(type(person_df))

train_id_df =  pd.read_csv('train_id.csv')
test_id_df =  pd.read_csv('test_id.csv')
validation_id_df =  pd.read_csv('validation_id.csv')


filtered_train_df = person_df[person_df['PERSON_ID'].isin(train_id_df['PERSON_ID'])]
filtered_test_df = person_df[person_df['PERSON_ID'].isin(test_id_df['PERSON_ID'])]
filtered_validation_df = person_df[person_df['PERSON_ID'].isin(validation_id_df['PERSON_ID'])]


# Write the DataFrame to a CSV file
filtered_train_df.to_csv('filtered_train_df.csv', index=False)
filtered_test_df.to_csv('filtered_test_df.csv', index=False)
filtered_validation_df.to_csv('filtered_validation_df.csv', index=False)


##################################################################################
###########                 Extracting demographic information
##################################################################################



train_persons_df = pd.read_csv('filtered_train_df.csv')
validation_persons_df = pd.read_csv('filtered_validation_df.csv')

# Combine training and validation data
combined_df = pd.concat([train_persons_df, validation_persons_df], ignore_index=True)

# Normalize the case for 'RACE_SOURCE_VALUE' and 'ETHNICITY_SOURCE_VALUE'
combined_df['RACE_SOURCE_VALUE'] = combined_df['RACE_SOURCE_VALUE'].str.lower()
combined_df['ETHNICITY_SOURCE_VALUE'] = combined_df['ETHNICITY_SOURCE_VALUE'].str.lower()

# Update 'RACE_SOURCE_VALUE' with 'Patient Refused' and 'Unknown'from 'ETHNICITY_SOURCE_VALUE'
ethnicity_values_to_move = ['patient refused', 'unknown']
mask = combined_df['ETHNICITY_SOURCE_VALUE'].isin(ethnicity_values_to_move)
combined_df.loc[mask, 'RACE_SOURCE_VALUE'] = combined_df.loc[mask, 'ETHNICITY_SOURCE_VALUE']

# Standardize the 'unknown' values in 'RACE_SOURCE_VALUE'
combined_df['RACE_SOURCE_VALUE'] = combined_df['RACE_SOURCE_VALUE'].replace({'unknown': 'Unknown', 'patient refused': 'Patient Refused'})

# Drop the 'ETHNICITY_SOURCE_VALUE' column
combined_df = combined_df.drop(columns=['ETHNICITY_SOURCE_VALUE'])

# Find unique types of race in combined data
unique_races_combined = combined_df['RACE_SOURCE_VALUE'].unique()
print("Unique types of race in combined data:", unique_races_combined)

# Count the number of persons with each type of race in combined data
race_counts_combined = combined_df['RACE_SOURCE_VALUE'].value_counts()


# Convert the counts to a DataFrame for better readability
race_counts_df_combined = race_counts_combined.reset_index()
race_counts_df_combined.columns = ['Race', 'Count']

print("\nRace counts in combined data:")
print(race_counts_df_combined)


# Calculate percentages for combined data
race_counts_df_combined['Percentage'] = race_counts_df_combined['Count'] / race_counts_df_combined['Count'].sum() * 100

# Plotting the pie chart for combined data without labels
plt.figure(figsize=(8, 6))
patches_combined, _ = plt.pie(race_counts_df_combined['Count'], labels=['']*len(race_counts_df_combined), startangle=140)
plt.title('Ethnicity Distribution (Train + Validation Data)')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Adding a legend with percentages and race types for combined data
legend_labels_combined = [f'{race} - {percentage:.1f}%' for race, percentage in zip(race_counts_df_combined['Race'], race_counts_df_combined['Percentage'])]
plt.legend(legend_labels_combined, loc='best', bbox_to_anchor=(1, 0.5))

plt.show()


# Read the test data
test_persons_df = pd.read_csv('filtered_test_df.csv')

# Normalize the case for 'RACE_SOURCE_VALUE' in test data
test_persons_df['RACE_SOURCE_VALUE'] = test_persons_df['RACE_SOURCE_VALUE'].str.lower()

# Standardize the 'unknown' values in 'RACE_SOURCE_VALUE' in test data
test_persons_df['RACE_SOURCE_VALUE'] = test_persons_df['RACE_SOURCE_VALUE'].replace({'unknown': 'Unknown', 'patient refused': 'Patient Refused'})

# Count the number of persons with each type of race in test data
race_counts_test = test_persons_df['RACE_SOURCE_VALUE'].value_counts()


# Convert the counts to a DataFrame for better readability
race_counts_df_test = race_counts_test.reset_index()
race_counts_df_test.columns = ['Race', 'Count']

print("\nRace counts in test data:")
print(race_counts_df_test)


# Calculate percentages for test data
race_counts_df_test['Percentage'] = race_counts_df_test['Count'] / race_counts_df_test['Count'].sum() * 100

# Plotting the pie chart for test data without labels
plt.figure(figsize=(8, 6))
patches_test, _ = plt.pie(race_counts_df_test['Count'], labels=['']*len(race_counts_df_test), startangle=140)
plt.title('Ethnicity Distribution (Test Data)')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Adding a legend with percentages and race types for test data
legend_labels_test = [f'{race} - {percentage:.1f}%' for race, percentage in zip(race_counts_df_test['Race'], race_counts_df_test['Percentage'])]
plt.legend(legend_labels_test, loc='best', bbox_to_anchor=(1, 0.5))

plt.show()



print(race_counts_df_test.to_string(index=False))
print(race_counts_df_combined.to_string(index=False))


