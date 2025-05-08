import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import random  # For generating random numbers


# define TimeGAN Model Components
def build_timegan_model(input_shape, latent_dim):
    # generator
    def build_generator():
        inputs = layers.Input(shape=(None, latent_dim))
        x = layers.LSTM(64, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.5)(x)  # Added Dropout
        x = layers.LSTM(64, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.5)(x)  # Added Dropout
        x = layers.Dense(input_shape[1])(x)
        return Model(inputs, x, name="generator")

    # discriminator
    def build_discriminator():
        inputs = layers.Input(shape=(None, input_shape[1]))
        x = layers.LSTM(64, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.5)(x)  # Added Dropout
        x = layers.LSTM(64, return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.5)(x)  # Added Dropout
        x = layers.Dense(1)(x)  # No activation function here
        return Model(inputs, x, name="discriminator")

    # autoencoder
    def build_autoencoder():
        inputs = layers.Input(shape=(None, input_shape[1]))
        x = layers.LSTM(64, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.LSTM(64, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dense(latent_dim)(x)
        return Model(inputs, x, name="autoencoder")

    generator = build_generator()
    discriminator = build_discriminator()
    autoencoder = build_autoencoder()

    return generator, discriminator, autoencoder


# prepare data
def prepare_data(data_folder_path, seq_len):
    sequences = []
    scaler = MinMaxScaler()

    for file_name in os.listdir(data_folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(data_folder_path, file_name)
            df = pd.read_csv(file_path)
            df['time'] = pd.to_datetime(df['time'])
            df = df[['value_temp', 'value_hum', 'value_acid']]

            # normalize data
            scaled_data = scaler.fit_transform(df)

            # create sequences
            def create_sequences(data, seq_len):
                seqs = []
                for i in range(len(data) - seq_len + 1):
                    seqs.append(data[i:i + seq_len])
                return np.array(seqs)

            sequences.append(create_sequences(scaled_data, seq_len))

    sequences = np.concatenate(sequences, axis=0)
    return sequences, scaler


# train TimeGAN
def train_timegan(data, epochs=200, batch_size=64):
    input_shape = data.shape[1:]
    latent_dim = 10

    generator, discriminator, autoencoder = build_timegan_model(input_shape, latent_dim)

    # compile models
    discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002), loss='binary_crossentropy')
    generator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002), loss='binary_crossentropy')
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002), loss='mse')

    # training loop
    for epoch in range(epochs):
        # train discriminator
        idx = np.random.randint(0, data.shape[0], batch_size)
        real_data = data[idx]
        noise = np.random.randn(batch_size, input_shape[0], latent_dim)
        fake_data = generator.predict(noise)

        # add noise to discriminator labels
        d_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_data, np.zeros((batch_size, 1)))

        # train generator
        noise = np.random.randn(batch_size, input_shape[0], latent_dim)
        g_loss = discriminator.train_on_batch(generator.predict(noise), np.ones((batch_size, 1)))

        # print progress
        if (epoch + 1) % 10 == 0:
            print(
                f'Epoch {epoch + 1}/{epochs}, D Loss Real: {d_loss_real:.4f}, D Loss Fake: {d_loss_fake:.4f}, G Loss: {g_loss:.4f}')

    return generator


# generate synthetic data
def generate_synthetic_data(generator, scaler, num_samples=5362, seq_len=10):
    latent_dim = 10
    generated_sequences = generator.predict(np.random.randn(num_samples // seq_len, seq_len, latent_dim))
    synthetic_data = scaler.inverse_transform(generated_sequences.reshape(-1, generated_sequences.shape[-1]))
    return synthetic_data


# main script
def main():
    script_dir = os.path.dirname(__file__)
    data_folder_path = os.path.join(script_dir, '..', 'correct_labeling', 'labeled_data')

    # generate a random number
    random_number = random.randint(1000, 99999)

    # create the output file name with the random number
    output_file = os.path.join(script_dir, f'generated_data_{random_number}.csv')

    seq_len = 10

    # prepare the data
    sequences, scaler = prepare_data(data_folder_path, seq_len)

    # train timeGAN
    generator = train_timegan(sequences, epochs=2000, batch_size=64)

    # generate synthetic data
    synthetic_data = generate_synthetic_data(generator, scaler, num_samples=8192, seq_len=seq_len)

    # save synthetic data
    synthetic_df = pd.DataFrame(synthetic_data, columns=['value_temp', 'value_hum', 'value_acid'])
    synthetic_df.to_csv(output_file, index=False)
    print(f'Synthetic data saved to {output_file}')


if __name__ == "__main__":
    main()
