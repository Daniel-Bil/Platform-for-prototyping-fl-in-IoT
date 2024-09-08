import pandas as pd
import os

# Define the output directory for forecasts
output_dir = './generated_data'

# Function to display the content of forecast files
def display_forecast_files(directory):
    for file_name in os.listdir(directory):
        if file_name.endswith('_FORECAST.csv'):
            file_path = os.path.join(directory, file_name)
            data = pd.read_csv(file_path)
            print(f"Contents of {file_path}:")
            print(data.head(), "\n")

# Display the forecast files
display_forecast_files(output_dir)
