import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 10, 'figure.autolayout': True})

DATA_DIR = os.path.join("data")
OUTPUT_DIR = os.path.join("data", "synthetic_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 🎛️ PARAMETRY BADAWCZE
SEQ_LEN = 144  # Długość sekwencji w krokach (144 * 10 min = 24 godziny)
LATENT_DIM = 8  # Wymiar przestrzeni ukrytej (kompresja)
EPOCHS = 50  # Liczba epok treningowych
BATCH_SIZE = 32


def create_sequences(data, seq_len):
    """Tworzy okna przesuwne (sekwencje) z danych szeregów czasowych."""
    sequences = []
    for i in range(len(data) - seq_len):
        sequences.append(data[i:i + seq_len])
    return np.array(sequences)


class Sampling(layers.Layer):
    """Warstwa wykonująca Reparameterization Trick: z = mu + sigma * epsilon"""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# NOWOŚĆ: Niestandardowa warstwa strat kompatybilna z Keras 3
class VAELossLayer(layers.Layer):
    """Opakowuje obliczenia błędu VAE w warstwę, by uniknąć błędu 'KerasTensor'"""

    def call(self, inputs, outputs, z_mean, z_log_var):
        # 1. Błąd rekonstrukcji (MSE)
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(tf.keras.losses.mse(inputs, outputs), axis=-1)
        )
        # 2. Dywergencja KL
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
        )
        self.add_loss(reconstruction_loss + kl_loss)
        return outputs


def build_lstm_vae(seq_len, num_features, latent_dim):
    """Buduje model Variational Autoencoder oparty na warstwach LSTM."""
    # ================= ENCODER =================
    encoder_inputs = layers.Input(shape=(seq_len, num_features), name='encoder_input')
    x = layers.LSTM(64, return_sequences=True, activation='tanh')(encoder_inputs)
    x = layers.LSTM(32, activation='tanh')(x)

    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

    z = Sampling()([z_mean, z_log_var])
    encoder = models.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')

    # ================= DECODER =================
    latent_inputs = layers.Input(shape=(latent_dim,), name='z_sampling')
    x = layers.RepeatVector(seq_len)(latent_inputs)
    x = layers.LSTM(32, return_sequences=True, activation='tanh')(x)
    x = layers.LSTM(64, return_sequences=True, activation='tanh')(x)
    decoder_outputs = layers.TimeDistributed(layers.Dense(num_features))(x)

    decoder = models.Model(latent_inputs, decoder_outputs, name='decoder')

    # ================= VAE =================
    z_mean, z_log_var, z = encoder(encoder_inputs)
    outputs = decoder(z)

    # Przepuszczenie przez warstwę liczącą błąd (zgodne z Keras 3)
    outputs = VAELossLayer()(encoder_inputs, outputs, z_mean, z_log_var)

    vae = models.Model(encoder_inputs, outputs, name='lstm_vae')
    vae.compile(optimizer='adam')

    return vae, encoder, decoder


def generate_synthetic_data(clean_csv_path):
    filename = os.path.basename(clean_csv_path)
    sensor_id = filename.replace('_CLEANED.csv', '')
    print(f"\n🚀 Rozpoczęto budowę cyfrowego bliźniaka dla: {sensor_id}")

    # 1. Wczytanie i przygotowanie danych
    df = pd.read_csv(clean_csv_path, index_col='time', parse_dates=True)
    features = ['value_temp', 'value_hum', 'value_acid', 'value_PV']
    df = df[features].dropna()  # Upewniamy się, że nie ma luk

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    X_train = create_sequences(scaled_data, SEQ_LEN)
    print(f"Przygotowano {X_train.shape[0]} sekwencji treningowych.")

    # 2. Budowa i trening modelu LSTM-VAE
    vae, encoder, decoder = build_lstm_vae(SEQ_LEN, len(features), LATENT_DIM)

    print("⏳ Trenowanie modelu VAE (uruchamiam na CPU)...")
    # History ignorujemy ostrzeżenie o braku 'y' (ponieważ liczymy loss z warstwy)
    history = vae.fit(X_train, X_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, verbose=1)

    # 3. Generowanie syntetycznych paczek danych
    num_synthetic_samples = 300  # Ile 24-godzinnych okien chcemy wygenerować
    random_latent_vectors = np.random.normal(size=(num_synthetic_samples, LATENT_DIM))

    synthetic_scaled = decoder.predict(random_latent_vectors)

    # 4. Sklejanie wygenerowanych sekwencji i odwrotna normalizacja
    synthetic_continuous = synthetic_scaled.reshape(-1, len(features))
    synthetic_real_scale = scaler.inverse_transform(synthetic_continuous)

    df_synthetic = pd.DataFrame(synthetic_real_scale, columns=features)

    # Tworzymy fikcyjny indeks czasowy, krok 10 minut
    start_date = pd.Timestamp("2024-01-01 00:00:00", tz='UTC')
    df_synthetic['time'] = pd.date_range(start=start_date, periods=len(df_synthetic), freq='10min')
    df_synthetic.set_index('time', inplace=True)

    out_csv = os.path.join(OUTPUT_DIR, f"{sensor_id}_SYNTHETIC.csv")
    df_synthetic.to_csv(out_csv)

    # ==========================================
    # WIZUALIZACJA: RZECZYWISTE vs SYNTETYCZNE
    # ==========================================
    samples_5days = 144 * 5
    real_sample = df.iloc[:samples_5days]
    synth_sample = df_synthetic.iloc[:samples_5days]

    fig, axes = plt.subplots(nrows=len(features), ncols=2, figsize=(18, 12))
    fig.suptitle(f'Digital Twin: {sensor_id} (Real vs Synthetic 5-Day Window)', fontsize=18, weight='bold')

    colors = ['#d62728', '#1f77b4', '#9467bd', '#17becf']

    for i, col in enumerate(features):
        c = colors[i % len(colors)]

        axes[i, 0].plot(range(len(real_sample)), real_sample[col], color=c)
        axes[i, 0].set_ylabel(col.replace('value_', '').upper(), weight='bold')
        axes[i, 0].set_title(f"Real Data Pattern ({col})")

        axes[i, 1].plot(range(len(synth_sample)), synth_sample[col], color=c, linestyle='--')
        axes[i, 1].set_title(f"Synthetic Twin Pattern ({col})")

    plt.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, f"{sensor_id}_synthetic_comparison.png")
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"✅ Zakończono! Zapisano dane: {out_csv} oraz wykres: {out_png}")


if __name__ == "__main__":
    clean_files = glob.glob(os.path.join(DATA_DIR, "*_CLEANED.csv"))
    if not clean_files:
        print(f"❌ Nie znaleziono plików *_CLEANED.csv w folderze {DATA_DIR}")
    else:
        # Rozpoczynamy od pierwszego znalezionego pliku
        process_target = clean_files[0]
        generate_synthetic_data(process_target)
        print("\n🎉 Generowanie danych syntetycznych zakończone.")