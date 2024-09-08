import pandas as pd
from prophet import Prophet
import os

# Define the output directory for forecasts
output_dir = './generated_data'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Expected column names
expected_columns = ['time', 'value_temp', 'value_hum', 'value_acid']


# Function to process each file
def process_file(file_path):
    try:
        # Load the generated data
        data = pd.read_csv(file_path)

        # Log the columns to diagnose issues
        print(f"Processing file: {file_path}")
        print(f"Columns in the loaded data: {data.columns.tolist()}")

        # Check if the columns match the expected columns
        if not set(expected_columns).issubset(data.columns):
            print(f"Required columns {expected_columns} not found in {file_path}")
            return False

        # Convert the 'time' column to datetime format and remove timezone
        data['time'] = pd.to_datetime(data['time']).dt.tz_localize(None)

        # Remove duplicates based on the 'time' column
        data = data.drop_duplicates(subset=['time'])

        # Set 'time' as the index
        data.set_index('time', inplace=True)

        # Ensure the time index has a frequency
        data = data.asfreq('min')

        # Interpolate the missing values
        data = data.interpolate(method='time')

        # Validate interpolation
        if data.isnull().values.any():
            print(f"Data still contains NaN values after interpolation for {file_path}")
            return False

        # Function to create and save forecasts
        def create_and_save_forecast(data, column_name, future_periods, file_suffix):
            series = data[[column_name]].reset_index()
            series.columns = ['ds', 'y']

            # Initialize the Prophet model with added daily and weekly seasonality
            model = Prophet()
            model.add_seasonality(name='daily', period=1, fourier_order=15)
            model.add_seasonality(name='weekly', period=7, fourier_order=10)

            # Fit the Prophet model
            model.fit(series)

            # Create future dataframe for specified periods with 10-minute interval
            future = model.make_future_dataframe(periods=future_periods, freq='10T')

            # Forecast
            forecast = model.predict(future)

            # Select the forecasted data
            forecast_df = forecast[['ds', 'yhat']].tail(future_periods)

            # Rename columns for saving
            forecast_df.columns = ['time', column_name]

            # Define the output file path
            output_forecast_path = file_path.replace('_GENERATED.csv', f'_{file_suffix}_FORECAST.csv')

            # Save the forecasted data to a new CSV file
            forecast_df.to_csv(output_forecast_path, index=False, header=True)

            print(f"Forecasted data for {column_name} saved to: {output_forecast_path}")

        # Forecast for temperature
        create_and_save_forecast(data, 'value_temp', future_periods=1000, file_suffix='TEMP')

        # Forecast for humidity
        create_and_save_forecast(data, 'value_hum', future_periods=1000, file_suffix='HUM')

        # Forecast for acidity
        create_and_save_forecast(data, 'value_acid', future_periods=1000, file_suffix='ACID')

        return True

    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
        return False


# Loop over each generated file in the output directory
for file_name in os.listdir(output_dir):
    if file_name.endswith('_GENERATED.csv') and os.path.isfile(os.path.join(output_dir, file_name)):
        file_path = os.path.join(output_dir, file_name)
        success = process_file(file_path)
        if not success:
            print(f"Error processing file: {file_path}")