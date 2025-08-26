import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


# Note:
# because the output layer uses sigmoid, inputs should be scaled [0, 1]
class AutoOutlier():

    # Builds a 1-hidden-layer autoencoder:
    # input(80) -> dense(32, relu) -> dense(80, sigmoid)
    # compiled with Adam (optimizer) + MSE (loss function)
    def __init__(self):
        self.input_size = 80
        self.encoder_input = layers.Input(shape=(self.input_size,), name='input_layer')
        # size of the encoded representation and the activation function can be adjusted per your liking
        self.encoded = layers.Dense(32, activation='relu', name='encoded_layer')(self.encoder_input)

        # decoder
        self.decoded = layers.Dense(self.input_size, activation='sigmoid', name='output_layer')(self.encoded)

        # autoencoder
        self.autoencoder = models.Model(self.encoder_input, self.decoded)
        self.autoencoder.compile(optimizer='adam', loss='mse')

        # summary of the autoencoder model
        self.autoencoder.summary()

    # unsupervised training for 50 epochs using the same dataset (samples) as inputs and targets
    def fit(self, samples):
        self.autoencoder.fit(samples, samples, epochs=50, batch_size=256, shuffle=True, validation_data=(samples, samples))

    # gets reconstruction with model.predict
    # computes per-row MSE reconstruction error
    # flags anomalies as those above the 95th percentile threshold, returning a boolean mask
    def reconstruct(self, samples):
        reconstructed_data = self.autoencoder.predict(samples)
        mse = np.mean(np.power(samples - reconstructed_data, 2), axis=1)
        threshold = np.quantile(mse, 0.95)
        anomalies = mse > threshold
        return anomalies

# instantiate the class (prints a summary)
if __name__ == "__main__":
    aut = AutoOutlier()