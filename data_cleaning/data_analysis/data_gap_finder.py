import os
import pandas as pd


def find_time_gaps(df, time_column='time', threshold_minutes=15):
    # Convert the time column to datetime
    df[time_column] = pd.to_datetime(df[time_column])

    # Ensure the dataframe is sorted by time
    df = df.sort_values(by=time_column)

    # Calculate the time differences between consecutive rows
    df['time_diff'] = df[time_column].diff().dt.total_seconds() / 60  # in minutes

    # Find where the time difference exceeds the threshold
    gaps_list = []
    for i in range(1, len(df)):
        if df['time_diff'].iloc[i] > threshold_minutes:
            start_gap = df[time_column].iloc[i - 1]
            end_gap = df[time_column].iloc[i]
            start_row = df.index[i - 1] + 1  # Adjusting for 1-based index
            end_row = df.index[i] + 1  # Adjusting for 1-based index
            gaps_list.append((start_row, start_gap, end_row, end_gap))

    return gaps_list


def process_csv_files_in_folder(folder_path, label_column='label', target_label='good', time_column='time',
                                threshold_minutes=15):
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if it's a CSV file
        if os.path.isfile(file_path) and filename.endswith('.csv'):
            # Load the CSV file into a DataFrame
            df = pd.read_csv(file_path)

            # Check if the label and time columns exist
            if label_column in df.columns and time_column in df.columns:
                print(f"Processing file: {filename}")

                # Identify gaps in time
                gaps = find_time_gaps(df, time_column=time_column, threshold_minutes=threshold_minutes)

                # Print the gaps with numbering
                if gaps:
                    print(f"File: {filename} - Found gaps:")
                    for idx, (start_row, start_gap, end_row, end_gap) in enumerate(gaps, start=1):
                        print(
                            f"  Gap {idx}: Start of gap: {start_gap} (Line {start_row}), End of gap: {end_gap} (Line {end_row})")
                else:
                    print(f"File: {filename} - No significant gaps found.")
            else:
                print(f"File: {filename} does not have the required columns.\n")


# Usage
script_dir = os.path.dirname(__file__)
data_folder_path = os.path.join(script_dir, '..', "labeling_data")
process_csv_files_in_folder(data_folder_path)