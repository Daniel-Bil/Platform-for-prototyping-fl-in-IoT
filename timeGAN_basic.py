import numpy as np
import pandas as pd
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
from ydata_synthetic.synthesizers.timeseries import TimeSeriesSynthesizer
# from ydata_synthetic.preprocessing.timeseries.utils import real_data_loading

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

if __name__ == "__main__":
    pass

    data_dir = Path("dane/segment_representation_4h_interpolated")
    feature_cols = ['value_temp', 'value_hum', 'value_acid', 'value_PV']
    sequence_length = 300  # adjust to your use case
    step = 10  # sliding window step
    sequences = []

    # Load and segment all files
    for csv_file in data_dir.rglob("*.csv"):
        df = pd.read_csv(csv_file)
        values = df[feature_cols].values
        if len(values) >= sequence_length:
            for i in range(0, len(values) - sequence_length + 1, step):
                seq = values[i:i + sequence_length]
                sequences.append(seq)

    data = np.array(sequences)  # shape: [samples, sequence_length, features]

    # Normalize per feature (global min-max across all samples)
    num_features = data.shape[2]
    scalers = []
    for i in range(num_features):
        scaler = MinMaxScaler()
        data[:, :, i] = scaler.fit_transform(data[:, :, i])
        scalers.append(scaler)

    # Define model parameters
    gan_args = ModelParameters(batch_size=128,
                               lr=5e-4,
                               noise_dim=32,
                               layers_dim=128,
                               latent_dim=24,
                               gamma=1)

    train_args = TrainParameters(epochs=100,
                                 sequence_length=sequence_length,
                                 number_sequences=len(data))

    synth = TimeSeriesSynthesizer(modelname="timegan", model_parameters=gan_args)
    synth.train(data, train_args)

    # Generate synthetic samples
    synthetic_data = synth.sample(len(data))

    # Choose an example index
    idx = 0
    real = data[idx]
    fake = synthetic_data[idx]

    plt.figure(figsize=(15, 6))
    for i, feature in enumerate(feature_cols):
        plt.subplot(2, 2, i + 1)
        plt.plot(real[:, i], label='Real')
        plt.plot(fake[:, i], label='Generated')
        plt.title(feature)
        plt.legend()
    plt.tight_layout()
    plt.show()