import os
import pandas as pd

# folders containing the input and output datasets
script_dir = os.path.dirname(__file__)
data_folder_path = os.path.join(script_dir, '..', 'timegan', 'corrected_data')
output_folder_path = os.path.join(script_dir, '..', 'timegan', 'labeled_data')  # Folder to save labeled data

# ensure the output folder exists
os.makedirs(output_folder_path, exist_ok=True)

# pre-calculated thresholds

thresholds = {
    'generated_data_15906.csv': {'temp': 7.40, 'hum': 23.04, 'acid': 0.30},
    'generated_data_1647.csv': {'temp': 6.24, 'hum': 30.25, 'acid': 0.32},
    'generated_data_20679.csv': {'temp': 12.96, 'hum': 24.16, 'acid': 0.26},
    'generated_data_24297.csv': {'temp': 6.39, 'hum': 22.95, 'acid': 0.30},
    'generated_data_2729.csv': {'temp': 6.18, 'hum': 25.35, 'acid': 0.33},
    'generated_data_33347.csv': {'temp': 5.84, 'hum': 23.05, 'acid': 0.28},
    'generated_data_34152.csv': {'temp': 6.61, 'hum': 18.62, 'acid': 0.37},
    'generated_data_38631.csv': {'temp': 6.12, 'hum': 28.78, 'acid': 0.31},
    'generated_data_43036.csv': {'temp': 5.91, 'hum': 23.62, 'acid': 0.28},
    'generated_data_47057.csv': {'temp': 7.01, 'hum': 21.38, 'acid': 0.55},
    'generated_data_48200.csv': {'temp': 5.63, 'hum': 16.45, 'acid': 0.31},
    'generated_data_53310.csv': {'temp': 5.82, 'hum': 21.12, 'acid': 0.29},
    'generated_data_60536.csv': {'temp': 10.32, 'hum': 19.09, 'acid': 0.43},
    'generated_data_66592.csv': {'temp': 5.67, 'hum': 19.60, 'acid': 0.32},
    'generated_data_69040.csv': {'temp': 5.57, 'hum': 21.32, 'acid': 0.59},
    'generated_data_71715.csv': {'temp': 6.85, 'hum': 25.89, 'acid': 0.40},
    'generated_data_72640.csv': {'temp': 5.29, 'hum': 20.37, 'acid': 0.40},
    'generated_data_76411.csv': {'temp': 7.02, 'hum': 28.24, 'acid': 0.23},
    'generated_data_77379.csv': {'temp': 5.61, 'hum': 20.81, 'acid': 0.29},
    'generated_data_83638.csv': {'temp': 10.24, 'hum': 18.10, 'acid': 0.35},
    'generated_data_8543.csv': {'temp': 8.01, 'hum': 34.05, 'acid': 0.52},
    'generated_data_86512.csv': {'temp': 5.08, 'hum': 29.97, 'acid': 0.27},
    'generated_data_87090.csv': {'temp': 6.23, 'hum': 31.49, 'acid': 0.28},
    'generated_data_92039.csv': {'temp': 5.59, 'hum': 20.45, 'acid': 0.33}
}



previous_thresholds = {
    'df_RuralIoT_001.csv': {'temp': 1.59, 'hum': 9.67, 'acid': 0.35},
    'df_RuralIoT_002.csv': {'temp': 0.85, 'hum': 10.82, 'acid': 0.52},
    'df_RuralIoT_003.csv': {'temp': 1.46, 'hum': 3.55, 'acid': 0.15},
    'df_RuralIoT_010.csv': {'temp': 1.10, 'hum': 23.72, 'acid': 0.38},
    'df_RuralIoT_21.csv': {'temp': 1.48, 'hum': 7.44, 'acid': 0.33},
    'df_RuralIoT_22.csv': {'temp': 1.99, 'hum': 29.25, 'acid': 2.96},
    'df_RuralIoT_23.csv': {'temp': 1.38, 'hum': 23.09, 'acid': 0.89}
}

# loop through all CSV files in the folder
for filename in os.listdir(data_folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(data_folder_path, filename)

        # load the dataset
        df = pd.read_csv(file_path)

        #df['time'] = pd.to_datetime(df['time'])

        # calculate the differences between consecutive rows
        temp_diff = df['value_temp'].diff().abs()
        hum_diff = df['value_hum'].diff().abs()
        acid_diff = df['value_acid'].diff().abs()

        # get the thresholds for the current dataset
        if filename in thresholds:
            temp_jump_threshold = thresholds[filename]['temp']
            hum_jump_threshold = thresholds[filename]['hum']
            acid_jump_threshold = thresholds[filename]['acid']
        else:
            print(f"Thresholds not found for {filename}. Skipping...")
            continue

        # add the label column with default value 'good'
        df['label'] = 'good'

        # identify indices where sudden jumps occur
        temp_error_indices = temp_diff[temp_diff > temp_jump_threshold].index
        hum_error_indices = hum_diff[hum_diff > hum_jump_threshold].index
        acid_error_indices = acid_diff[acid_diff > acid_jump_threshold].index

        # combine all error indices into a set
        error_indices = set(temp_error_indices) | set(hum_error_indices) | set(acid_error_indices)

        # label only the row where the sudden jump is detected
        for idx in error_indices:
            df.at[idx, 'label'] = 'error'

        # handle missing data (gaps) by marking the appropriate rows as 'error'
        # define the gap threshold (e.g., 15 minutes)
        gap_threshold = pd.Timedelta(minutes=15)
        #df['time_diff'] = df['time'].diff()
        #gap_indices = df[df['time_diff'] > gap_threshold].index

        # label only the row where the gap is detected
        #for idx in gap_indices:
        #    df.at[idx, 'label'] = 'error'

        # remove intermediate columns before saving
        #df = df.drop(columns=['time_diff'])

        # save the labeled dataset to a new CSV file
        labeled_file_path = os.path.join(output_folder_path, filename)
        df.to_csv(labeled_file_path, index=False)

        # print the thresholds for the current dataset
        print(f"Dataset: {filename}")
        print(f"Temperature jump threshold: {temp_jump_threshold:.2f}°C")
        print(f"Humidity jump threshold: {hum_jump_threshold:.2f}%")
        print(f"Acidity jump threshold: {acid_jump_threshold:.2f} pH units")
        print("-" * 40)
