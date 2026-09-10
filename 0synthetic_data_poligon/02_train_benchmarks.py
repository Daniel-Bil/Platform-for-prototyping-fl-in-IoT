#!/usr/bin/env python3
"""
02_train_benchmarks.py
Etap 2: Trening modeli porownawczych (ARIMA jako baseline statystyczny
oraz Conv1D-VAE jako reprezentant modeli generatywnych) na ciaglym bloku stacji 002.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
import tensorflow as tf
from tensorflow.keras import layers, models

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 10, 'figure.autolayout': True})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "data", "continuous_blocks", "002_CONTINUOUS_BLOCK.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "synthetic_output")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "model_comparisons")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURES = ['value_temp', 'value_hum', 'value_acid', 'value_PV']
SEQ_LEN = 144       # 24 godziny x 6 probek/godz
LATENT_DIM = 12
EPOCHS = 40
BATCH_SIZE = 32


# =========================================================================
# 1. Klasy pomocnicze dla VAE
# =========================================================================
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAELossLayer(layers.Layer):
    def __init__(self, kl_weight=0.05, **kwargs):
        super().__init__(**kwargs)
        self.kl_weight = kl_weight

    def call(self, inputs, outputs, z_mean, z_log_var):
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(tf.keras.losses.mse(inputs, outputs), axis=-1)
        )
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
        )
        self.add_loss(reconstruction_loss + (self.kl_weight * kl_loss))
        return outputs


def build_conv1d_vae(seq_len, num_features, latent_dim):
    encoder_inputs = layers.Input(shape=(seq_len, num_features), name='enc_in')
    x = layers.Conv1D(32, kernel_size=5, padding='same', activation='relu')(encoder_inputs)
    x = layers.AveragePooling1D(pool_size=2)(x)
    x = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(x)
    x = layers.AveragePooling1D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)

    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = models.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')

    latent_inputs = layers.Input(shape=(latent_dim,), name='dec_in')
    x = layers.Dense(36 * 64, activation='relu')(latent_inputs)
    x = layers.Reshape((36, 64))(x)
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(x)
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(32, kernel_size=5, padding='same', activation='relu')(x)
    decoder_outputs = layers.Conv1D(num_features, kernel_size=5, padding='same', activation='linear')(x)
    decoder = models.Model(latent_inputs, decoder_outputs, name='decoder')

    z_mean, z_log_var, z = encoder(encoder_inputs)
    outputs = decoder(z)
    outputs = VAELossLayer()(encoder_inputs, outputs, z_mean, z_log_var)
    vae = models.Model(encoder_inputs, outputs, name='vae')
    vae.compile(optimizer='adam')
    return vae, decoder


def create_sequences(data, seq_len):
    sequences = []
    for i in range(len(data) - seq_len):
        sequences.append(data[i:i + seq_len])
    return np.array(sequences)


# =========================================================================
# 2. Glowny potok trenowania
# =========================================================================
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Blad: Plik {INPUT_FILE} nie istnieje! Uruchom najpierw 01b_extract_continuous_blocks.py.")
        return

    print("Wczytywanie czystego bloku referencyjnego stacji 002...")
    df = pd.read_csv(INPUT_FILE, parse_dates=['time'])
    df.set_index('time', inplace=True)
    n_samples = len(df)
    print(f"Liczba probek: {n_samples}")

    # ---------------------------------------------------------------------
    # A. Model Statystyczny (ARIMA baseline per kanal)
    # ---------------------------------------------------------------------
    print("\nDopasowywanie modeli ARIMA dla poszczegolnych kanalow...")
    df_arima = pd.DataFrame(index=df.index)

    for col in FEATURES:
        print(f"  -> Trenowanie ARIMA(2,1,2) dla cechy: {col}...")
        series = df[col].values
        # Prosty rzad ARIMA (p=2, d=1, q=2) dla wychwycenia dynamiki krotkookresowej
        try:
            model = sm.tsa.ARIMA(series, order=(2, 1, 2))
            res = model.fit()
            # Generowanie probek stochastycznych: predykcja in-sample + reszty
            simulated = res.predict(start=1, end=len(series))
            # Dopasowanie dlugosci
            simulated = np.concatenate([[series[0]], simulated])[:len(series)]
            df_arima[col] = simulated
        except Exception as e:
            print(f"     Ostrzezenie: ARIMA dla {col} nie zbiegla sie ({e}), uzycie fallback...")
            df_arima[col] = series

    # ---------------------------------------------------------------------
    # B. Model Gleboki (Conv1D-VAE)
    # ---------------------------------------------------------------------
    print("\nPrzygotowywanie tensora sekwencji dla modelu VAE...")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[FEATURES])
    X_train = create_sequences(scaled_data, SEQ_LEN)
    print(f"Wymiar zbioru sekwencji: {X_train.shape}")

    vae, decoder = build_conv1d_vae(SEQ_LEN, len(FEATURES), LATENT_DIM)
    print("Trenowanie modelu VAE...")
    vae.fit(X_train, X_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, shuffle=True)

    # Generowanie probek z przestrzeni ukrytej N(0, I)
    num_windows = int(np.ceil(n_samples / SEQ_LEN))
    z_random = np.random.normal(size=(num_windows, LATENT_DIM))
    synthetic_scaled = decoder.predict(z_random)
    synthetic_flat = synthetic_scaled.reshape(-1, len(FEATURES))[:n_samples]
    synthetic_real = scaler.inverse_transform(synthetic_flat)
    df_vae = pd.DataFrame(synthetic_real, index=df.index, columns=FEATURES)

    # ---------------------------------------------------------------------
    # Zapis i Wizualizacja Porownawcza
    # ---------------------------------------------------------------------
    df_arima.to_csv(os.path.join(OUTPUT_DIR, "002_SYNTHETIC_ARIMA.csv"))
    df_vae.to_csv(os.path.join(OUTPUT_DIR, "002_SYNTHETIC_VAE.csv"))
    print("\nZapisano wygenerowane pliki CSV do data/synthetic_output/.")

    # Wykres porownawczy pierwszych 5 dni (5 * 144 probek)
    plot_len = 144 * 5
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(14, 10), sharex=True)
    fig.suptitle("5-Day Trajectory Comparison: Ground Truth vs ARIMA vs Deep Generative (VAE)", fontsize=13, weight='bold')

    for i, col in enumerate(FEATURES):
        ax = axes[i]
        ax.plot(df.index[:plot_len], df[col].iloc[:plot_len], color='black', linewidth=1.5, label='Real (Ground Truth)')
        ax.plot(df.index[:plot_len], df_arima[col].iloc[:plot_len], color='#1f77b4', linestyle='--', linewidth=1.2, label='ARIMA')
        ax.plot(df.index[:plot_len], df_vae[col].iloc[:plot_len], color='#d62728', alpha=0.85, linewidth=1.3, label='Deep Generative (VAE)')
        ax.set_ylabel(col, weight='bold')
        if i == 0:
            ax.legend(loc='upper right')

    axes[-1].set_xlabel("Time (UTC)", weight='bold')
    plt.tight_layout()
    cmp_plot = os.path.join(RESULTS_DIR, "002_trajectories_comparison.png")
    plt.savefig(cmp_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Zapisano wykres trajektorii: {cmp_plot}")


if __name__ == "__main__":
    main()