#!/usr/bin/env python3
"""
02b_train_timegan.py
Etap 2b: Trening modelu TimeGAN (Yoon et al., 2019) na ciaglym bloku stacji 002.
Architektura oparta na sieciach GRU: Embedding, Recovery, Generator, Discriminator.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "data", "continuous_blocks", "002_CONTINUOUS_BLOCK.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "synthetic_output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES = ['value_temp', 'value_hum', 'value_acid', 'value_PV']
SEQ_LEN = 144       # 24h (okno dobowe)
HIDDEN_DIM = 24     # Wymiar wektora ukrytego GRU
EPOCHS = 100        # Liczba epok dla faz pre-training oraz joint
BATCH_SIZE = 64


def create_sequences(data, seq_len):
    sequences = []
    for i in range(len(data) - seq_len):
        sequences.append(data[i:i + seq_len])
    return np.array(sequences, dtype=np.float32)


# =========================================================================
# Moduly sieciowe TimeGAN (Embedding, Recovery, Generator, Discriminator)
# =========================================================================
def build_embedding_network(seq_len, num_features, hidden_dim):
    x_in = layers.Input(shape=(seq_len, num_features))
    h = layers.GRU(hidden_dim, return_sequences=True, activation='tanh')(x_in)
    h = layers.GRU(hidden_dim, return_sequences=True, activation='tanh')(h)
    h_out = layers.Dense(hidden_dim, activation='sigmoid')(h)
    return models.Model(x_in, h_out, name='Embedder')


def build_recovery_network(seq_len, num_features, hidden_dim):
    h_in = layers.Input(shape=(seq_len, hidden_dim))
    r = layers.GRU(hidden_dim, return_sequences=True, activation='tanh')(h_in)
    r = layers.GRU(hidden_dim, return_sequences=True, activation='tanh')(r)
    x_out = layers.Dense(num_features, activation='sigmoid')(r)
    return models.Model(h_in, x_out, name='Recovery')


def build_generator_network(seq_len, num_features, hidden_dim):
    z_in = layers.Input(shape=(seq_len, num_features))
    g = layers.GRU(hidden_dim, return_sequences=True, activation='tanh')(z_in)
    g = layers.GRU(hidden_dim, return_sequences=True, activation='tanh')(g)
    e_out = layers.Dense(hidden_dim, activation='sigmoid')(g)
    return models.Model(z_in, e_out, name='Generator')


def build_discriminator_network(seq_len, hidden_dim):
    h_in = layers.Input(shape=(seq_len, hidden_dim))
    d = layers.GRU(hidden_dim, return_sequences=True, activation='tanh')(h_in)
    d = layers.GRU(hidden_dim, return_sequences=True, activation='tanh')(d)
    y_out = layers.Dense(1, activation='linear')(d)
    return models.Model(h_in, y_out, name='Discriminator')


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Blad: Plik {INPUT_FILE} nie istnieje!")
        return

    print("Wczytywanie danych stacji 002 pod trening TimeGAN...")
    df = pd.read_csv(INPUT_FILE, parse_dates=['time']).set_index('time')
    n_samples = len(df)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[FEATURES])
    X = create_sequences(scaled_data, SEQ_LEN)
    n_seq, seq_len, num_feat = X.shape
    print(f"Przygotowano tensor: {X.shape}")

    dataset = tf.data.Dataset.from_tensor_slices(X).shuffle(buffer_size=1024).batch(BATCH_SIZE)

    # Inicjalizacja komponentow
    embedder = build_embedding_network(seq_len, num_feat, HIDDEN_DIM)
    recovery = build_recovery_network(seq_len, num_feat, HIDDEN_DIM)
    generator = build_generator_network(seq_len, num_feat, HIDDEN_DIM)
    discriminator = build_discriminator_network(seq_len, HIDDEN_DIM)

    opt_e = optimizers.Adam(learning_rate=0.001)
    opt_r = optimizers.Adam(learning_rate=0.001)
    opt_s = optimizers.Adam(learning_rate=0.001)
    opt_g = optimizers.Adam(learning_rate=0.001)
    opt_d = optimizers.Adam(learning_rate=0.001)

    # ---------------------------------------------------------------------
    # Faza 1: Pre-training Autoenkodera (Embedding + Recovery)
    # ---------------------------------------------------------------------
    print("\n--- Faza 1: Pre-training Autoenkodera (Rekonstrukcja) ---")
    mse_loss = tf.keras.losses.MeanSquaredError()

    for epoch in range(1, EPOCHS + 1):
        loss_epoch = []
        for batch_x in dataset:
            with tf.GradientTape() as tape:
                h = embedder(batch_x)
                x_tilde = recovery(h)
                e_loss = 10.0 * tf.sqrt(mse_loss(batch_x, x_tilde))

            trainable_vars = embedder.trainable_variables + recovery.trainable_variables
            grads = tape.gradient(e_loss, trainable_vars)
            opt_e.apply_gradients(zip(grads, trainable_vars))
            loss_epoch.append(float(e_loss))

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoka {epoch}/{EPOCHS} | E_Loss: {np.mean(loss_epoch):.4f}")

    # ---------------------------------------------------------------------
    # Faza 2: Supervised Step-Loss Training (Generator uczy sie dynamiki H)
    # ---------------------------------------------------------------------
    print("\n--- Faza 2: Supervised Loss Training (Dynamika czasowa) ---")
    for epoch in range(1, EPOCHS + 1):
        loss_epoch = []
        for batch_x in dataset:
            z_batch = tf.random.uniform(shape=[tf.shape(batch_x)[0], seq_len, num_feat], minval=0.0, maxval=1.0)
            with tf.GradientTape() as tape:
                h = embedder(batch_x)
                h_hat = generator(z_batch)
                # Strata krokowa supervised
                s_loss = mse_loss(h[:, 1:, :], h_hat[:, :-1, :])

            grads = tape.gradient(s_loss, generator.trainable_variables)
            opt_s.apply_gradients(zip(grads, generator.trainable_variables))
            loss_epoch.append(float(s_loss))

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoka {epoch}/{EPOCHS} | S_Loss: {np.mean(loss_epoch):.4f}")

    # ---------------------------------------------------------------------
    # Faza 3: Joint Adversarial Training (Pelen trening adwersarialny)
    # ---------------------------------------------------------------------
    print("\n--- Faza 3: Pelny trening adwersarialny TimeGAN ---")
    bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    for epoch in range(1, EPOCHS + 1):
        d_losses, g_losses = [], []
        for batch_x in dataset:
            cur_batch_size = tf.shape(batch_x)[0]
            z_batch = tf.random.uniform(shape=[cur_batch_size, seq_len, num_feat], minval=0.0, maxval=1.0)

            # Trening dyskryminatora
            with tf.GradientTape() as tape_d:
                h = embedder(batch_x)
                h_hat = generator(z_batch)

                y_real = discriminator(h)
                y_fake = discriminator(h_hat)

                d_loss_real = bce_loss(tf.ones_like(y_real), y_real)
                d_loss_fake = bce_loss(tf.zeros_like(y_fake), y_fake)
                d_loss = d_loss_real + d_loss_fake

            grads_d = tape_d.gradient(d_loss, discriminator.trainable_variables)
            opt_d.apply_gradients(zip(grads_d, discriminator.trainable_variables))
            d_losses.append(float(d_loss))

            # Trening generatora i embeddera
            with tf.GradientTape() as tape_g:
                h = embedder(batch_x)
                h_hat = generator(z_batch)
                x_hat = recovery(h_hat)

                y_fake = discriminator(h_hat)

                # Strata nienadzorowana GAN
                g_loss_u = bce_loss(tf.ones_like(y_fake), y_fake)
                # Strata nadzorowana
                g_loss_s = mse_loss(h[:, 1:, :], h_hat[:, :-1, :])
                # Momenty (srednia i wariancja)
                g_loss_v = tf.reduce_mean(tf.abs(tf.sqrt(tf.nn.moments(x_hat, [0])[1] + 1e-6) -
                                                tf.sqrt(tf.nn.moments(batch_x, [0])[1] + 1e-6)))

                g_loss = g_loss_u + 100.0 * tf.sqrt(g_loss_s) + 100.0 * g_loss_v

            g_vars = generator.trainable_variables + embedder.trainable_variables
            grads_g = tape_g.gradient(g_loss, g_vars)
            opt_g.apply_gradients(zip(grads_g, g_vars))
            g_losses.append(float(g_loss))

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoka {epoch}/{EPOCHS} | D_Loss: {np.mean(d_losses):.4f} | G_Loss: {np.mean(g_losses):.4f}")

    # ---------------------------------------------------------------------
    # Generowanie syntetycznego szeregu i odwracanie skali
    # ---------------------------------------------------------------------
    print("\nGenerowanie probek TimeGAN...")
    num_windows = int(np.ceil(n_samples / SEQ_LEN))
    z_samples = tf.random.uniform(shape=[num_windows, seq_len, num_feat], minval=0.0, maxval=1.0)
    h_hat_samples = generator(z_samples)
    x_hat_samples = recovery(h_hat_samples).numpy()

    synth_flat = x_hat_samples.reshape(-1, num_feat)[:n_samples]
    synth_real = scaler.inverse_transform(synth_flat)

    df_timegan = pd.DataFrame(synth_real, index=df.index, columns=FEATURES)
    out_csv = os.path.join(OUTPUT_DIR, "002_SYNTHETIC_TIMEGAN.csv")
    df_timegan.to_csv(out_csv)
    print(f"Zapisano wygenerowany zbiór TimeGAN: {out_csv}")


if __name__ == "__main__":
    main()