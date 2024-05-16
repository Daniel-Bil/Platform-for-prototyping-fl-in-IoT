import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import collections
from logic.dataProcesing import preprocess

# Adjusted Data Preprocessing
def preprocess_error_data(samples):
    # Flatten the input data
    flattened_samples = samples.reshape(samples.shape[0], -1)
    # Assuming binary classification labels, e.g., 0 for no error, 1 for error
    labels = np.random.randint(0, 2, size=(flattened_samples.shape[0], 1)).astype(np.float32)
    return flattened_samples, labels

# Custom model for error finding
class ErrorFindingModel(tff.learning.models.VariableModel):
    def __init__(self):
        self._keras_model = Sequential([
            Dense(64, activation='relu', input_shape=(90,)),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        self._keras_model.compile(optimizer='adam', loss='binary_crossentropy')
        self._loss_object = tf.keras.losses.BinaryCrossentropy()
        self._metrics = [tf.keras.metrics.BinaryAccuracy()]

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
            'x': tf.TensorSpec(shape=(None, 90), dtype=tf.float32),
            'y': tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
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
        return collections.OrderedDict({
            metric.name: lambda metric_result: tf.reduce_mean(metric_result) for metric in self._metrics
        })

    def report_local_unfinalized_metrics(self):
        return collections.OrderedDict(
            binary_accuracy=[metric.result() for metric in self._metrics])

    def reset_metrics(self):
        for metric in self._metrics:
            metric.reset_states()

# Function to provide an instance of the error finding model
def error_model_fn():
    return ErrorFindingModel()

# Adjusted create_federated_data for error finding
def create_federated_data_error(samples, labels, num_clients=10):
    client_datasets = []
    num_samples_per_client = len(samples) // num_clients
    for i in range(num_clients):
        start = i * num_samples_per_client
        end = (i + 1) * num_samples_per_client

        x = samples[start:end].astype(np.float32)
        y = labels[start:end].astype(np.float32)

        dataset = tf.data.Dataset.from_tensor_slices({'x': x, 'y': y}).batch(5)
        client_datasets.append(dataset)

    return client_datasets

if __name__ == '__main__':
    # Using the preprocess function to obtain normalized sliding windows (samples)
    samples = preprocess()

    # Preprocess data for error finding model
    flattened_samples, labels = preprocess_error_data(samples)

    # Create the federated datasets for error finding
    federated_train_data = create_federated_data_error(flattened_samples, labels)

    # Initialize the federated averaging process for error finding model
    iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
        error_model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=0.001),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
    )

    state = iterative_process.initialize()

    # Train the error finding model
    for round_num in range(1, 11):
        state, metrics = iterative_process.next(state, federated_train_data)
        print(f'Error Finding Round {round_num}, metrics={metrics}')
