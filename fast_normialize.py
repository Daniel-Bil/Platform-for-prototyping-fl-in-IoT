import os
import pandas as pd
import numpy as np

# Define fixed ranges for humidity
HUMIDITY_MIN, HUMIDITY_MAX = 0, 100  # Fixed range for humidity

def normalize_column(column, col_min, col_max):
    """Normalize a column with min-max normalization."""
    return (column - col_min) / (col_max - col_min)

def compute_global_min_max(directory_path):
    """Compute the global min and max for value_temp and value_acid across all files."""
    global_min_max = {
        'value_temp_min': float('inf'),
        'value_temp_max': float('-inf'),
        'value_acid_min': float('inf'),
        'value_acid_max': float('-inf')
    }

    # Traverse through all CSV files to compute global min and max
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(directory_path, file_name)
            df = pd.read_csv(file_path)

            # Update global min and max for value_temp and value_acid
            global_min_max['value_temp_min'] = min(global_min_max['value_temp_min'], df['value_temp'].min())
            global_min_max['value_temp_max'] = max(global_min_max['value_temp_max'], df['value_temp'].max())
            global_min_max['value_acid_min'] = min(global_min_max['value_acid_min'], df['value_acid'].min())
            global_min_max['value_acid_max'] = max(global_min_max['value_acid_max'], df['value_acid'].max())

    return global_min_max

def process_file(file_path, output_dir, global_min_max):
    """Normalize a single file using global min and max values."""
    # Load the data
    df = pd.read_csv(file_path)

    # Normalize value_hum using fixed min and max (0 to 100)
    df['value_hum'] = normalize_column(df['value_hum'], HUMIDITY_MIN, HUMIDITY_MAX)

    # Normalize value_temp and value_acid using global min-max scaling
    df['value_temp'] = normalize_column(df['value_temp'], global_min_max['value_temp_min'], global_min_max['value_temp_max'])
    df['value_acid'] = normalize_column(df['value_acid'], global_min_max['value_acid_min'], global_min_max['value_acid_max'])

    # Create output file path
    file_name = os.path.basename(file_path).replace('.csv', '_normalized.csv')
    output_file_path = os.path.join(output_dir, file_name)

    # Save the normalized file
    df.to_csv(output_file_path, index=False)

def normalize_files_in_directory(directory_path, output_directory):
    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # First pass: compute global min and max values across all files
    global_min_max = compute_global_min_max(directory_path)
    print(f"Global Min-Max values: {global_min_max}")

    # Second pass: Normalize each file using the global min-max values
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(directory_path, file_name)
            process_file(file_path, output_directory, global_min_max)






if __name__ == "__main__":

    # Directory paths
    input_directory = "dane\\generated_data"  # Replace with the correct path to your directory
    output_directory = "dane\\normalized_data"

    # Run the normalization process
    normalize_files_in_directory(input_directory, output_directory)

    print("Normalization process completed.")








