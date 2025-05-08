import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta


def generate_random_seconds_variations(base_seconds, num_samples):
    """Generates random seconds variations around a base delta."""
    variations = []
    for _ in range(num_samples):
        # random variation in seconds (like in originals): between -10 and +10 seconds
        variation = np.random.randint(-10, 10)
        variations.append(base_seconds + variation)
    return variations


def add_timestamps(df, start_time, base_seconds=600):
    """Adds timestamps to the dataframe, starting from 'start_time' with a base delta of 'base_seconds'."""
    num_rows = len(df)

    # generate base delta as timedelta
    base_delta = timedelta(seconds=base_seconds)

    # create a list of timestamps
    timestamps = [start_time]
    random_seconds = generate_random_seconds_variations(base_seconds, num_rows - 1)

    for seconds in random_seconds:
        next_time = timestamps[-1] + timedelta(seconds=seconds)
        timestamps.append(next_time)

    # convert timestamps to original format: 'YYYY-MM-DDTHH:MM:SSZ'
    formatted_timestamps = [timestamp.strftime('%Y-%m-%dT%H:%M:%SZ') for timestamp in timestamps]

    # insert timestamps as the first column
    df.insert(0, 'time', formatted_timestamps)
    return df


script_dir = os.path.dirname(__file__)
input_folder = os.path.join(script_dir, '..', 'timegan', 'corrected_data')
output_folder = os.path.join(script_dir, '..', 'timegan', 'data_with_timestamps')

# create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# process each file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_folder, filename)

        # load dataset
        df = pd.read_csv(file_path)

        # define start time (this is just an example; adjust as needed)
        start_time = datetime.strptime('2023-04-13T06:09:14Z', '%Y-%m-%dT%H:%M:%SZ')

        # add timestamps to the dataframe
        df_with_timestamps = add_timestamps(df, start_time)

        # save the updated dataframe to a new CSV file
        new_file_path = os.path.join(output_folder, filename)
        df_with_timestamps.to_csv(new_file_path, index=False)

        print(f"Timestamps added and file saved as '{new_file_path}'.")

print("All files processed.")
