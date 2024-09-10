import os
import pandas as pd

def calculate_statistics(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Convert columns to numeric, forcing errors to NaN
    df = df.apply(pd.to_numeric, errors='coerce')

    # Calculate statistics for temperature, humidity, and acidity
    stats = {
        'File': os.path.basename(file_path),
        'Temperature': {
            'min': df['value_temp'].min(),
            'max': df['value_temp'].max(),
            'mean': df['value_temp'].mean(),
            'std': df['value_temp'].std()
        },
        'Humidity': {
            'min': df['value_hum'].min(),
            'max': df['value_hum'].max(),
            'mean': df['value_hum'].mean(),
            'std': df['value_hum'].std()
        },
        'Acidity': {
            'min': df['value_acid'].min(),
            'max': df['value_acid'].max(),
            'mean': df['value_acid'].mean(),
            'std': df['value_acid'].std()
        }
    }
    return stats

def process_all_csvs(folder_path="."):
    all_stats = []

    # Process each CSV file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            try:
                stats = calculate_statistics(file_path)
                all_stats.append(stats)

                # Print the statistics for each file
                print(f"File: {stats['File']}")
                print(f"  Temperature - min: {stats['Temperature']['min']:.4f}, max: {stats['Temperature']['max']:.4f}, mean: {stats['Temperature']['mean']:.4f}, std: {stats['Temperature']['std']:.4f}")
                print(f"  Humidity    - min: {stats['Humidity']['min']:.4f}, max: {stats['Humidity']['max']:.4f}, mean: {stats['Humidity']['mean']:.4f}, std: {stats['Humidity']['std']:.4f}")
                print(f"  Acidity     - min: {stats['Acidity']['min']:.4f}, max: {stats['Acidity']['max']:.4f}, mean: {stats['Acidity']['mean']:.4f}, std: {stats['Acidity']['std']:.4f}")
                print()
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

if __name__ == "__main__":
    process_all_csvs(".")  # Process CSVs in the current folder
