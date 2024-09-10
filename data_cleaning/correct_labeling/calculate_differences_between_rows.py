import os
import pandas as pd
import numpy as np

# for datasets from professor Bogdan
#script_dir = os.path.dirname(__file__)
#data_folder_path = os.path.join(script_dir, '..', "..", 'dane')


# for time gan
script_dir = os.path.dirname(__file__)
data_folder_path = os.path.join(script_dir, '..', 'timegan', 'corrected_data')


# Loop through all CSV files in the folder
for filename in os.listdir(data_folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(data_folder_path, filename)

        # Load the dataset
        df = pd.read_csv(file_path)

        # Ensure the time column is parsed as datetime
        #df['time'] = pd.to_datetime(df['time']) # uncomment later

        # Calculate the differences between consecutive rows
        df['temp_diff'] = df['value_temp'].diff().abs()
        df['hum_diff'] = df['value_hum'].diff().abs()
        df['acid_diff'] = df['value_acid'].diff().abs()

        # For each parameter, calculate the mean and standard deviation of the differences
        temp_mean = df['temp_diff'].mean()
        temp_std = df['temp_diff'].std()
        hum_mean = df['hum_diff'].mean()
        hum_std = df['hum_diff'].std()
        acid_mean = df['acid_diff'].mean()
        acid_std = df['acid_diff'].std()

        # Calculate the thresholds for sudden jumps (mean + 3 * std)
        temp_jump_threshold = temp_mean + 3 * temp_std
        hum_jump_threshold = hum_mean + 3 * hum_std
        acid_jump_threshold = acid_mean + 3 * acid_std

        # Print the calculated thresholds for the current dataset
        print(f"Dataset: {filename}")
        print(f"Temperature jump threshold: {temp_jump_threshold:.2f}°C")
        print(f"Humidity jump threshold: {hum_jump_threshold:.2f}%")
        print(f"Acidity jump threshold: {acid_jump_threshold:.2f} pH units")
        print("-" * 40)
