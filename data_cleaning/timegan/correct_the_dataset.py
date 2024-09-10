import pandas as pd
import os


def shift_negative_to_positive(df):
    df['value_temp'] = df['value_temp'] - df['value_temp'].min()
    df['value_hum'] = df['value_hum'] - df['value_hum'].min()
    df['value_acid'] = df['value_acid'] - df['value_acid'].min()
    df['value_hum'] = df['value_hum'].clip(lower=0)
    return df


def scale_values(df, temp_min, temp_max, hum_min, hum_max, acid_min, acid_max):
    # Scaling Temperature
    temp_range = temp_max - temp_min
    df['value_temp'] = df['value_temp'] / df['value_temp'].max() * temp_range + temp_min

    # Scaling Humidity
    hum_range = hum_max - hum_min
    df['value_hum'] = df['value_hum'] / df['value_hum'].max() * hum_range + hum_min

    # Scaling Acidity
    acid_range = acid_max - acid_min
    df['value_acid'] = df['value_acid'] / df['value_acid'].max() * acid_range + acid_min

    return df


def process_files(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    for file in files:
        file_path = os.path.join(input_folder, file)
        df = pd.read_csv(file_path)

        # Convert and scale
        df = shift_negative_to_positive(df)
        df = scale_values(df, temp_min=0, temp_max=30, hum_min=0, hum_max=100, acid_min=2.5, acid_max=4.0)

        # Save to new file
        output_path = os.path.join(output_folder, file)
        df.to_csv(output_path, index=False)

        # Print file name
        print(f'Processed file saved as: {output_path}')


# Directories
input_folder = '.'
output_folder = 'corrected_data'

# Process the files
process_files(input_folder, output_folder)
