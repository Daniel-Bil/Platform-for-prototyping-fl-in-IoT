import os
import pandas as pd


def process_csv_files_in_folder(folder_path, label_column='label', target_label='good'):
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if it's a CSV file
        if os.path.isfile(file_path) and filename.endswith('.csv'):
            # Load the CSV file into a DataFrame
            df = pd.read_csv(file_path)

            # Check if the label column exists
            if label_column in df.columns:
                # Count occurrences of the target label
                total_count = len(df)
                good_count = (df[label_column] == target_label).sum()

                # Calculate percentage of the target label
                good_percentage = (good_count / total_count) * 100 if total_count > 0 else 0

                # Print results for the file
                print(f"File: {filename}")
                print(f"'{target_label}' labels: {good_count}")
                print(f"Percentage of '{target_label}' labels: {good_percentage:.2f}%\n")
            else:
                print(f"File: {filename} does not have a '{label_column}' column.\n")


script_dir = os.path.dirname(__file__)
data_folder_path = os.path.join(script_dir, '..', "labeling_data")
process_csv_files_in_folder(data_folder_path)