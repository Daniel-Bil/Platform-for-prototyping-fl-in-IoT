import pandas as pd
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(__file__)
data_file_path = os.path.join(script_dir, '..', 'dane', 'df_RuralIoT_001.csv')
# Load the data
df = pd.read_csv(data_file_path)

# Basic statistics
print(df['value_hum'].describe())

# Plot the distribution
plt.hist(df['value_hum'], bins=50)
plt.xlabel('Humidity')
plt.ylabel('Frequency')
plt.title('Distribution of Humidity Values')
plt.show()
