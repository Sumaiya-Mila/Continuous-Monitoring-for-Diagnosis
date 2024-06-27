import pandas as pd
import matplotlib.pyplot as plt

# Read the training and validation data
train_persons_df = pd.read_csv('filtered_train_df.csv')
validation_persons_df = pd.read_csv('filtered_validation_df.csv')

# Combine training and validation data
combined_df = pd.concat([train_persons_df, validation_persons_df], ignore_index=True)

# Normalize the case for 'GENDER_SOURCE_VALUE'
combined_df['GENDER_SOURCE_VALUE'] = combined_df['GENDER_SOURCE_VALUE'].str.lower()

# Find unique types of gender in combined data
unique_genders_combined = combined_df['GENDER_SOURCE_VALUE'].unique()
print("Unique types of gender in combined data:", unique_genders_combined)

# Count the number of persons with each type of gender in combined data
gender_counts_combined = combined_df['GENDER_SOURCE_VALUE'].value_counts()
print("\nNumber of persons with each type of gender in combined data:")
print(gender_counts_combined)

# Convert the counts to a DataFrame for better readability
gender_counts_df_combined = gender_counts_combined.reset_index()
gender_counts_df_combined.columns = ['Gender', 'Count']

# Calculate percentages for combined data
gender_counts_df_combined['Percentage'] = gender_counts_df_combined['Count'] / gender_counts_df_combined['Count'].sum() * 100

# Display gender distribution in combined data as a table
print("\nGender distribution in Combined Data:")
print(gender_counts_df_combined.to_string(index=False))

# Read the test data
test_persons_df = pd.read_csv('filtered_test_df.csv')

# Normalize the case for 'GENDER_SOURCE_VALUE' in test data
test_persons_df['GENDER_SOURCE_VALUE'] = test_persons_df['GENDER_SOURCE_VALUE'].str.lower()

# Count the number of persons with each type of gender in test data
gender_counts_test = test_persons_df['GENDER_SOURCE_VALUE'].value_counts()
print("\nNumber of persons with each type of gender in test data:")
print(gender_counts_test)

# Convert the counts to a DataFrame for better readability
gender_counts_df_test = gender_counts_test.reset_index()
gender_counts_df_test.columns = ['Gender', 'Count']

# Calculate percentages for test data
gender_counts_df_test['Percentage'] = gender_counts_df_test['Count'] / gender_counts_df_test['Count'].sum() * 100

# Display gender distribution in test data as a table
print("\nGender distribution in Test Data:")
print(gender_counts_df_test.to_string(index=False))
