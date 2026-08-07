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

SEQ_LEN = 144
LATENT_DIM = 12  # Zwiększamy nieco pojemność mózgu sieci
EPOCHS = 50
BATCH_SIZE = 32


def create_sequences(data, seq_len):
    sequences = []
    for i in range(len(data) - seq_len):
        sequences.append(data[i:i + seq_len])
    return np.array(sequences)


class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAELossLayer(layers.Layer):
    def call(self, inputs, outputs, z_mean, z_log_var):
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(tf.keras.losses.mse(inputs, outputs), axis=-1)
        )

        # Wprowadzamy KL Weight, aby sieć nie ignorowała kształtu danych (Anti-Collapse)
        kl_weight = 0.05
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
        )

        self.add_loss(reconstruction_loss + (kl_weight * kl_loss))
        return outputs


def build_conv1d_vae(seq_len, num_features, latent_dim):
    """Buduje model Conv1D-VAE, wybitnie radzący sobie z generowaniem gładkich sygnałów."""
    # ================= ENCODER (Kompresja fali) =================
    encoder_inputs = layers.Input(shape=(seq_len, num_features), name='encoder_input')

    x = layers.Conv1D(32, kernel_size=5, padding='same', activation='relu')(encoder_inputs)
    x = layers.AveragePooling1D(pool_size=2)(x)  # Redukcja: 144 -> 72

    x = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(x)
    x = layers.AveragePooling1D(pool_size=2)(x)  # Redukcja: 72 -> 36

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)

    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])

    encoder = models.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')

    # ================= DECODER (Rysowanie fali) =================
    latent_inputs = layers.Input(shape=(latent_dim,), name='z_sampling')

    x = layers.Dense(36 * 64, activation='relu')(latent_inputs)
    x = layers.Reshape((36, 64))(x)

    x = layers.UpSampling1D(size=2)(x)  # Rozszerzenie: 36 -> 72
    x = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(x)

    x = layers.UpSampling1D(size=2)(x)  # Rozszerzenie: 72 -> 144
    x = layers.Conv1D(32, kernel_size=5, padding='same', activation='relu')(x)

    # Liniowa aktywacja na końcu, by swobodnie operować w znormalizowanej przestrzeni 0-1
    decoder_outputs = layers.Conv1D(num_features, kernel_size=5, padding='same', activation='linear')(x)

    decoder = models.Model(latent_inputs, decoder_outputs, name='decoder')

    # ================= VAE =================
    z_mean, z_log_var, z = encoder(encoder_inputs)
    outputs = decoder(z)

    outputs = VAELossLayer()(encoder_inputs, outputs, z_mean, z_log_var)
    vae = models.Model(encoder_inputs, outputs, name='conv1d_vae')
    vae.compile(optimizer='adam')

    return vae, encoder, decoder


def generate_synthetic_data(clean_csv_path):
    filename = os.path.basename(clean_csv_path)
    sensor_id = filename.replace('_CLEANED.csv', '')
    print(f"\n🚀 Rozpoczęto budowę CONV1D-VAE dla: {sensor_id}")

    df = pd.read_csv(clean_csv_path, index_col='time', parse_dates=True)
    features = ['value_temp', 'value_hum', 'value_acid', 'value_PV']
    df = df[features].dropna()

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    X_train = create_sequences(scaled_data, SEQ_LEN)
    print(f"Przygotowano {X_train.shape[0]} sekwencji treningowych.")

    vae, encoder, decoder = build_conv1d_vae(SEQ_LEN, len(features), LATENT_DIM)

    print("⏳ Trenowanie modelu VAE...")
    history = vae.fit(X_train, X_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, verbose=1)

    # Generowanie syntetycznych paczek danych
    num_synthetic_samples = 300
    random_latent_vectors = np.random.normal(size=(num_synthetic_samples, LATENT_DIM))

    synthetic_scaled = decoder.predict(random_latent_vectors)

    synthetic_continuous = synthetic_scaled.reshape(-1, len(features))
    synthetic_real_scale = scaler.inverse_transform(synthetic_continuous)

    df_synthetic = pd.DataFrame(synthetic_real_scale, columns=features)

    start_date = pd.Timestamp("2024-01-01 00:00:00", tz='UTC')
    df_synthetic['time'] = pd.date_range(start=start_date, periods=len(df_synthetic), freq='10min')
    df_synthetic.set_index('time', inplace=True)

    # Zabezpieczenie fizyczne (dla pewności, by sieć nie wygenerowała wilgotności > 100)
    limits = {'value_temp': (-30, 60), 'value_hum': (0, 100), 'value_acid': (0, 14), 'value_PV': (0, 10)}
    for col in features:
        df_synthetic[col] = df_synthetic[col].clip(lower=limits[col][0], upper=limits[col][1])

    out_csv = os.path.join(OUTPUT_DIR, f"{sensor_id}_SYNTHETIC.csv")
    df_synthetic.to_csv(out_csv)

    # Wizualizacja
    samples_5days = 144 * 5
    real_sample = df.iloc[:samples_5days]
    synth_sample = df_synthetic.iloc[:samples_5days]

    fig, axes = plt.subplots(nrows=len(features), ncols=2, figsize=(18, 12))
    fig.suptitle(f'Digital Twin: {sensor_id} (Conv1D Real vs Synthetic)', fontsize=18, weight='bold')

    colors = ['#d62728', '#1f77b4', '#9467bd', '#17becf']

    for i, col in enumerate(features):
        c = colors[i % len(colors)]

        axes[i, 0].plot(range(len(real_sample)), real_sample[col], color=c)
        axes[i, 0].set_ylabel(col.replace('value_', '').upper(), weight='bold')
        axes[i, 0].set_title(f"Real Data ({col})")

        # Alpha 0.85 dla syntetyku, żeby wyglądał naturalnie
        axes[i, 1].plot(range(len(synth_sample)), synth_sample[col], color=c, alpha=0.85)
        axes[i, 1].set_title(f"Synthetic Twin ({col})")

    plt.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, f"{sensor_id}_synthetic_comparison.png")
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"✅ Zakończono! Zapisano dane i wykres: {out_png}")


if __name__ == "__main__":
    clean_files = glob.glob(os.path.join(DATA_DIR, "*_CLEANED.csv"))
    if not clean_files:
        print(f"❌ Nie znaleziono plików *_CLEANED.csv w folderze {DATA_DIR}")
    else:
        for file in clean_files:
            generate_synthetic_data(file)
        print("\n🎉 Generowanie danych syntetycznych dla wszystkich czujników zakończone.")