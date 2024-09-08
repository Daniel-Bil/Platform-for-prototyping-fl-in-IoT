import pandas as pd
import os

# Define paths relative to the script location
script_dir = os.path.dirname(__file__)
data_folder_path = os.path.join(script_dir, '..', "..", 'dane')
output_folder_path = script_dir


# Funkcja do oznaczania błędów
def label_errors(df):
    errors = []

    for i in range(1, len(df)):
        prev_row = df.iloc[i - 1]
        curr_row = df.iloc[i]

        if curr_row['value_temp'] < -15 or curr_row['value_temp'] > 50:
            errors.append(i)
        if curr_row['value_acid'] > 9.5 or curr_row['value_acid'] < 3:
            errors.append(i)
        if curr_row['value_hum'] < 7.5 or curr_row['value_hum'] > 90:
            errors.append(i)

        # Inne reguły można dodać tutaj

    return errors


# Iterate through all files in the 'dane' folder
for filename in os.listdir(data_folder_path):
    if filename.endswith('.csv'):
        # Construct full file path
        data_file_path = os.path.join(data_folder_path, filename)

        # Read the data
        df = pd.read_csv(data_file_path)

        # Label the errors
        error_indices = label_errors(df)
        df['label'] = ['error' if i in error_indices else 'good' for i in range(len(df))]

        # Save the labeled data to a new file with "_labeled" appended to the filename
        output_file_name = f"{filename.split('.csv')[0]}_labeled.csv"
        output_file_path = os.path.join(output_folder_path, output_file_name)
        df.to_csv(output_file_path, index=False)

print("Labeling completed for all files.")
