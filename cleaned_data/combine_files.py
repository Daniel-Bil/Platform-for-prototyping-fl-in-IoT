import pandas as pd
import os

# Define the output directory for forecasts
output_dir = './generated_data'


# Function to combine forecast files for each original file
def combine_forecasts_for_each_file(directory):
    # Loop over each file in the output directory
    for file_name in os.listdir(directory):
        if file_name.endswith('_TEMP_FORECAST.csv'):
            base_name = file_name.replace('_TEMP_FORECAST.csv', '')
            temp_file = os.path.join(directory, f"{base_name}_TEMP_FORECAST.csv")
            hum_file = os.path.join(directory, f"{base_name}_HUM_FORECAST.csv")
            acid_file = os.path.join(directory, f"{base_name}_ACID_FORECAST.csv")

            # Ensure all three forecast files exist
            if os.path.isfile(temp_file) and os.path.isfile(hum_file) and os.path.isfile(acid_file):
                # Load the forecast data
                temp_df = pd.read_csv(temp_file)
                hum_df = pd.read_csv(hum_file)
                acid_df = pd.read_csv(acid_file)

                # Merge the dataframes on the 'time' column
                combined_df = temp_df.merge(hum_df, on='time').merge(acid_df, on='time')

                # Rename columns appropriately
                combined_df.columns = ['time', 'value_temp', 'value_hum', 'value_acid']

                # Define the output file path
                output_combined_path = os.path.join(directory, f"{base_name}_COMBINED_FORECAST.csv")

                # Save the combined forecast data to a new CSV file
                combined_df.to_csv(output_combined_path, index=False, header=True)

                print(f"Combined forecast data saved to: {output_combined_path}")


# Call the function to combine forecast files for each original file
combine_forecasts_for_each_file(output_dir)