import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


class AutoOutlier():
    def __init__(self):
        self.input_size = 80
        self.encoder_input = layers.Input(shape=(self.input_size,), name='input_layer')
        # You can adjust the size of the encoded representation and the activation function
        self.encoded = layers.Dense(32, activation='relu', name='encoded_layer')(self.encoder_input)

        # Decoder
        self.decoded = layers.Dense(self.input_size, activation='sigmoid', name='output_layer')(self.encoded)

        # Autoencoder
        self.autoencoder = models.Model(self.encoder_input, self.decoded)
        self.autoencoder.compile(optimizer='adam', loss='mse')

        # Summary of the autoencoder model
        self.autoencoder.summary()

    def fit(self, samples):
        self.autoencoder.fit(samples, samples, epochs=50, batch_size=256, shuffle=True, validation_data=(samples, samples))

    def reconstruct(self, samples):
        reconstructed_data = self.autoencoder.predict(samples)
        mse = np.mean(np.power(samples - reconstructed_data, 2), axis=1)
        threshold = np.quantile(mse, 0.95)
        anomalies = mse > threshold
        return anomalies