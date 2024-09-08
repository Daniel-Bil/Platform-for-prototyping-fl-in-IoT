import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import os

# Check if a GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the path to your CSV files
data_path = "../labeling_data/"

# Get all CSV files in the folder
csv_files = [file for file in os.listdir(data_path) if file.endswith('.csv')]

# Load and concatenate all CSV files into one DataFrame
df_list = [pd.read_csv(os.path.join(data_path, file)) for file in csv_files]
df = pd.concat(df_list, ignore_index=True)

# Extract time-based features
df['timestamp'] = pd.to_datetime(df['time'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_year'] = df['timestamp'].dt.dayofyear

# Select only the relevant columns
features = ['value_temp', 'value_hum', 'value_acid', 'value_PV']

# Normalize the feature columns (excluding labels)
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# Convert labels to numerical format
df['label'] = df['label'].map({'error': 0, 'good': 1})

# Separate data into features and labels
X_train = df[features].values
y_train = df['label'].values

# Convert data to Tensor and move to GPU
X_train = torch.Tensor(X_train).to(device)
y_train = torch.Tensor(y_train).to(device)  # Move labels to GPU too

# Define the Generator network
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, output_dim),
            nn.Tanh()  # Ensure outputs are between -1 and 1
        )

    def forward(self, z):
        return self.model(z)

# Define the Discriminator network
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Outputs probability of being real
        )

    def forward(self, x):
        return self.model(x)

# Hyperparameters
input_dim = len(features)  # Number of features (including time features)
latent_dim = 100  # Dimension of random noise
batch_size = 64
epochs = 150
lr = 0.0002

# Initialize networks and move them to GPU
generator = Generator(input_dim=latent_dim, output_dim=input_dim).to(device)
discriminator = Discriminator(input_dim=input_dim).to(device)

# Optimizers
optim_G = optim.Adam(generator.parameters(), lr=lr)
optim_D = optim.Adam(discriminator.parameters(), lr=lr)

# Loss function
criterion = nn.BCELoss()

# Training loop
for epoch in range(epochs):
    for i in range(0, len(X_train), batch_size):
        # Prepare real data
        real_data = X_train[i:i + batch_size].to(device)  # Ensure data is on GPU
        real_labels = torch.ones(real_data.size(0), 1).to(device)  # Move labels to GPU

        # Train Discriminator on real data
        optim_D.zero_grad()
        output_real = discriminator(real_data)
        loss_real = criterion(output_real, real_labels)

        # Generate fake data
        z = torch.randn(real_data.size(0), latent_dim).to(device)  # Noise to GPU
        fake_data = generator(z)
        fake_labels = torch.zeros(real_data.size(0), 1).to(device)  # Fake labels to GPU

        # Train Discriminator on fake data
        output_fake = discriminator(fake_data.detach())
        loss_fake = criterion(output_fake, fake_labels)

        # Backpropagate the loss for Discriminator
        loss_D = loss_real + loss_fake
        loss_D.backward()
        optim_D.step()

        # Train Generator
        optim_G.zero_grad()
        output_fake = discriminator(fake_data)
        loss_G = criterion(output_fake, real_labels)  # Generator wants to fool Discriminator
        loss_G.backward()
        optim_G.step()

    if epoch % 1 == 0:
        print(f"Epoch [{epoch}/{epochs}], Loss D: {loss_D.item()}, Loss G: {loss_G.item()}")

# Generate new synthetic data
z = torch.randn(1000, latent_dim).to(device)
synthetic_data = generator(z).detach().cpu().numpy()  # Move back to CPU for saving

# Rescale the synthetic data back to the original range
synthetic_data = scaler.inverse_transform(synthetic_data)

# Generate time series with seasonal and diurnal patterns
def generate_time_series(start_time, num_points, freq='10T'):
    """ Generate a list of timestamps starting from `start_time` with a frequency `freq`. """
    times = [start_time + timedelta(minutes=i*10) for i in range(num_points)]
    return [t.strftime('%Y-%m-%dT%H:%M:%SZ') for t in times]

# Create timestamps
start_time = datetime(2023, 4, 4, 19, 4, 6)  # Example start time
num_points = synthetic_data.shape[0]
timestamps = generate_time_series(start_time, num_points)

# Define the columns list
columns = ['value_temp', 'value_hum', 'value_acid', 'value_PV']

# Convert to DataFrame
synthetic_df = pd.DataFrame(synthetic_data, columns=columns)

# Add timestamps to DataFrame
synthetic_df['time'] = timestamps

# Reorder columns to have 'time' as the first column
synthetic_df = synthetic_df[['time'] + columns]

# Save synthetic data to CSV
synthetic_df.to_csv('synthetic_data.csv', index=False)

print("Synthetic data generation complete.")
