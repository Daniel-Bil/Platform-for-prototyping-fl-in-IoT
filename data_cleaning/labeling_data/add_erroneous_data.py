import pandas as pd
import numpy as np
import os


def introduce_errors(df, error_columns, error_probability=0.05, increase_range=(50, 200)):
    """Introduces random large errors into specified columns with a given probability."""
    df_with_errors = df.copy()

    for column in error_columns:
        # For each row, roll for a chance to apply an error
        for i in range(len(df_with_errors)):
            if np.random.rand() < error_probability:  # 5% chance to apply error
                # Increase the value in the specified column by a random percentage (50% to 200%)
                increase_percentage = np.random.uniform(increase_range[0], increase_range[1]) / 100
                df_with_errors.loc[i, column] *= (1 + increase_percentage)  # Apply the large increase

    return df_with_errors


script_dir = os.path.dirname(__file__)
input_folder = os.path.join(script_dir, '..', 'timegan', 'data_with_timestamps')
output_folder = os.path.join(script_dir, '..', 'timegan', 'data_with_large_errors')

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Define the columns where you want to introduce errors
error_columns = ['value_temp', 'value_hum', 'value_acid']  # Adjust as necessary

# Process each file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_folder, filename)

        # Load dataset
        df = pd.read_csv(file_path)

        # Introduce large errors to the dataframe
        df_with_errors = introduce_errors(df, error_columns)

        # Save the updated dataframe to a new CSV file in the output folder
        new_file_path = os.path.join(output_folder, filename)
        df_with_errors.to_csv(new_file_path, index=False)

        print(f"Large errors introduced and file saved as '{new_file_path}'.")

print("All files processed.")
