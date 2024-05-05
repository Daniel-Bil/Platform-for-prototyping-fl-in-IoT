import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import collections

import tensorflow as tf
import tensorflow_federated as tff
from logic.dataProcesing import preprocess
#TODO: Comment the code
#TODO: Go over the code and edit it
#TODO: Make a completely new model for error finding in data

# Custom model class implementing VariableModel requirements
class CustomLSTMModel(tff.learning.models.VariableModel):
    def __init__(self):
        # Create the Keras LSTM model
        self._keras_model = Sequential([
            LSTM(50, input_shape=(30, 3), return_sequences=False),
            Dense(3)
        ])
        self._keras_model.compile(optimizer='adam', loss='mean_squared_error')
        self._loss_object = tf.keras.losses.MeanSquaredError()
        self._metrics = [tf.keras.metrics.MeanSquaredError()]

        # Initialize variables
        self._variables = {
            'weights': self._keras_model.trainable_weights,
            'metrics': [m.variables for m in self._metrics]
        }

    @property
    def trainable_variables(self):
        return self._variables['weights']

    @property
    def non_trainable_variables(self):
        return []

    @property
    def local_variables(self):
        return self._variables['metrics']

    @property
    def input_spec(self):
        return {
            'x': tf.TensorSpec(shape=(None, 30, 3), dtype=tf.float32),
            'y': tf.TensorSpec(shape=(None, 3), dtype=tf.float32)
        }

    @tf.function
    def forward_pass(self, batch_input, training=True):
        x, y = batch_input['x'], batch_input['y']
        predictions = self._keras_model(x, training=training)
        loss = self._loss_object(y, predictions)
        num_examples = tf.shape(x)[0]
        for metric in self._metrics:
            metric.update_state(y, predictions)
        return tff.learning.models.BatchOutput(
            loss=loss, predictions=predictions, num_examples=num_examples)

    def predict_on_batch(self, x, training=False):
        return self._keras_model(x, training=training)

    def metric_finalizers(self):
        # Use OrderedDict to maintain consistent ordering
        return collections.OrderedDict({
            metric.name: lambda metric_result: tf.reduce_mean(metric_result) for metric in self._metrics
        })
    def report_local_unfinalized_metrics(self):
        return collections.OrderedDict(
            mean_squared_error=[metric.result() for metric in self._metrics])

    def reset_metrics(self):
        for metric in self._metrics:
            metric.reset_states()

# Function to provide an instance of the custom model
def model_fn():
    return CustomLSTMModel()

def create_federated_data(samples, num_clients=10):
    client_datasets = []
    num_samples_per_client = len(samples) // num_clients
    for i in range(num_clients):
        start = i * num_samples_per_client
        end = (i + 1) * num_samples_per_client

        # Cast NumPy arrays to float32 for TFF compatibility
        x = samples[start:end, :, :].astype(np.float32)
        y = samples[start:end, -1, :].astype(np.float32)

        # Create the dataset using TensorFlow
        dataset = tf.data.Dataset.from_tensor_slices({'x': x, 'y': y}).batch(5)
        client_datasets.append(dataset)

    return client_datasets


if __name__ == '__main__':
    # Using the preprocess function to obtain normalized sliding windows (samples)
    samples = preprocess()

    #print(samples)

    # Create the federated datasets
    federated_train_data = create_federated_data(samples)

    # Initialize the federated averaging process
    iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=0.001),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
    )

    state = iterative_process.initialize()

    for round_num in range(1, 11):
        state, metrics = iterative_process.next(state, federated_train_data)
        print(f'Round {round_num}, metrics={metrics}')
