from datetime import timedelta
import pandas as pd
from prophet import Prophet
import os

def generate_time_series(start_time, num_points, freq='10T'):
    """Generate a list of timestamps starting from `start_time` with a frequency `freq`."""
    times = [start_time + timedelta(minutes=i*10) for i in range(num_points)]
    return [t.strftime('%Y-%m-%dT%H:%M:%SZ') for t in times]

# Define the path to your CSV files
data_path = "../labeling_data/"

# Get all CSV files in the folder
csv_files = [file for file in os.listdir(data_path) if file.endswith('.csv')]

# Load and concatenate all CSV files into one DataFrame
df_list = [pd.read_csv(os.path.join(data_path, file)) for file in csv_files]
df = pd.concat(df_list, ignore_index=True)

# Extract time-based features
df['timestamp'] = pd.to_datetime(df['time']).dt.tz_localize(None)  # Remove timezone information
df['hour'] = df['timestamp'].dt.hour
df['day_of_year'] = df['timestamp'].dt.dayofyear

# Select only the relevant columns
features = ['value_temp', 'value_hum', 'value_acid', 'value_PV']
df_features = df[features].copy()

# Fill missing values with a forward fill
df_features.fillna(method='ffill', inplace=True)

# Prepare data for Prophet
prophet_df = pd.DataFrame({
    'ds': df['timestamp'],
    'y': df['value_PV']  # Target variable for prediction
})

# Initialize Prophet model with seasonality components
model = Prophet(
    yearly_seasonality=True,
    daily_seasonality=True,
    weekly_seasonality=True,
    seasonality_mode='additive'
)

# Add additional regressors for temperature, humidity, and acidity
model.add_regressor('value_temp')
model.add_regressor('value_hum')
model.add_regressor('value_acid')

# Fit the model with additional regressors
prophet_df = prophet_df.join(df_features[['value_temp', 'value_hum', 'value_acid']], how='left')
model.fit(prophet_df)

# Make future dataframe
future = model.make_future_dataframe(periods=365*4, freq='10T')  # Adjust periods and frequency as needed

# Add additional regressors to future dataframe
future = future.join(df_features[['value_temp', 'value_hum', 'value_acid']], how='left')

# Fill missing values in the future dataframe
future.fillna(method='ffill', inplace=True)

# Predict
forecast = model.predict(future)

# Save forecast to CSV
forecast.to_csv('forecast_results.csv', index=False)

print("Forecast generation complete. Results saved to forecast_results.csv.")
