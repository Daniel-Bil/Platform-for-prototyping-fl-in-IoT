import pandas as pd
import os

# Define the input and output directories
input_dir = '.'
output_dir = './generated_data'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Loop over each file in the input directory
for file_name in os.listdir(input_dir):
    if file_name.endswith('_CLEAN.csv') and os.path.isfile(os.path.join(input_dir, file_name)):
        file_path = os.path.join(input_dir, file_name)

        # Load the data
        data = pd.read_csv(file_path)

        # Convert the 'time' column to datetime format
        data['time'] = pd.to_datetime(data['time'])

        # Set 'time' as the index
        data.set_index('time', inplace=True)

        # Expected interval in minutes
        expected_interval = 10

        # Find the gaps
        time_diff = data.index.to_series().diff()
        large_gaps = time_diff > pd.Timedelta(minutes=expected_interval)

        # Generate missing timestamps within gaps
        all_times = data.index.tolist()
        for gap_start, is_large_gap in zip(data.index[:-1], large_gaps[1:]):
            if is_large_gap:
                gap_end = data.index[data.index.get_loc(gap_start) + 1]
                gap_times = pd.date_range(start=gap_start, end=gap_end, freq=f'{expected_interval}T')[1:]
                all_times.extend(gap_times)

        # Create a new DataFrame with the complete index
        all_times = pd.to_datetime(sorted(set(all_times)))
        complete_data = data.reindex(all_times)

        # Interpolate the missing values
        interpolated_data = complete_data.interpolate(method='time')

        # Reset index to get 'time' back as a column
        interpolated_data.reset_index(inplace=True)
        interpolated_data.rename(columns={'index': 'time'}, inplace=True)

        # Define the output file path
        output_file_path = os.path.join(output_dir, file_name.replace('_CLEAN', '_GENERATED'))

        # Save the interpolated data to the new CSV file
        interpolated_data.to_csv(output_file_path, index=False)

        print(f"Interpolated data saved to: {output_file_path}")
