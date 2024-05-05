import collections
import os

from copy import copy, deepcopy
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow.keras.layers import TimeDistributed

from logic.wrappers import time_wrapper
from scipy.signal import savgol_filter
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.nonparametric.smoothers_lowess import lowess
from pykalman import KalmanFilter

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from logic.Outlier_detectors import outlier_detector


#define for the app
NUM_CLIENTS = 10
# number of times entire dataset is passed through the model
NUM_EPOCHS = 5
# how many samples in one batch
BATCH_SIZE = 40
# size of buffer used for shuffling the data
SHUFFLE_BUFFER = 100
# how many batches prefetched
PREFETCH_BUFFER = 10
@time_wrapper
def find_interrupts_withPV(data: dict | pd.DataFrame) -> dict:

        number_of_measurements = len(data["time"])
        idxs = []
        for i in range(number_of_measurements):
            pvd = data['value_PV'][i] - data['value_PV'][i + 1] if i < number_of_measurements - 1 else 0
            print(pvd)
            if abs(pvd) > 3:
                idxs.append(i)
        print(idxs)
        start = 0

        good = []
        for z, idx in enumerate(idxs):
            time = data['time'][start:idx]
            value_temp = data['value_temp'][start:idx]
            value_hum = data['value_hum'][start:idx]
            value_acid = data['value_acid'][start:idx]
            value_PV = data['value_PV'][start:idx]

            good_course = {"time": time,
                           "value_temp": value_temp,
                           "value_hum": value_hum,
                           "value_acid": value_acid,
                           "value_PV": value_PV}
            good.append(good_course)
            start = idx+1
        return good


@time_wrapper
def find_interrupts_withTime(data: dict | pd.DataFrame, idx_only=None):
    '''
    Function that finds shift in time series and divides into list of good timeseries
    :param data:
    :return: list
    '''
    print(data["time"][:20])
    number_of_measurements = len(data["time"])
    idxs = []
    for i in range(number_of_measurements-1):
        one = datetime.fromisoformat(data["time"][i].replace('Z', '+00:00'))
        two = datetime.fromisoformat(data["time"][i + 1].replace('Z', '+00:00'))
        if int((two - one).total_seconds()) / 60 > 30:
            idxs.append(i)
    if idx_only is not None:
        return idxs
    print(idxs)
    start = 0

    good = []
    for z, idx in enumerate(idxs):
        idx = idx+1
        time = data['time'][start:idx]
        value_temp = data['value_temp'][start:idx]
        value_hum = data['value_hum'][start:idx]
        value_acid = data['value_acid'][start:idx]
        value_PV = data['value_PV'][start:idx]

        good_course = {"time": time,
                       "value_temp": value_temp,
                       "value_hum": value_hum,
                       "value_acid": value_acid,
                       "value_PV": value_PV}
        good.append(good_course)
        start = idx
    return good

@time_wrapper
def find_shift_in_timeseries(data1, data2) -> int:
    avg1 = []
    for j in range(len(data1["time"]) - 1):
        one = datetime.fromisoformat(data1["time"][j].replace('Z', '+00:00'))
        two = datetime.fromisoformat(data1["time"][j + 1].replace('Z', '+00:00'))
        avg1.append(int((two - one).total_seconds()) / 60)

    time = np.mean(avg1)

    start = datetime.fromisoformat(data1["time"][-1].replace('Z', '+00:00'))
    end = datetime.fromisoformat(data2["time"][0].replace('Z', '+00:00'))

    difference_in_minutes = int((end - start).total_seconds() / 60)
    return int(difference_in_minutes / time)

@time_wrapper
def normalize(data1:dict, data2:dict) -> (dict, dict):
    for key in ["value_temp","value_hum","value_acid","value_PV"]:

        max1: float = max([max(data1[key]), max(data2[key])])
        min1: float = min([min(data1[key]), min(data2[key])])
        data1[key] = [(d-min1)/(max1-min1) for d in data1[key]]
        data2[key] = [(d-min1)/(max1-min1) for d in data2[key]]
    return data1, data2

def denormalize(data:np.array, min_values:dict, max_values:dict) -> (np.array):

    denorm_data = np.zeros_like(data)

    for i, key in enumerate(['value_temp', 'value_hum', 'value_acid']):
        min_val = min_values[key]
        max_val = max_values[key]
        denorm_data[:, i] = data[:, i] * (max_val - min_val) + min_val
    return denorm_data


# (data is a dictionary of arrays)
def create_basic_data(data: dict):
    size_of_sample = 30
    number_of_samples = len(data['value_temp']) - size_of_sample + 1
    samples = []
    for i in range(number_of_samples):
        idx = i
        d1 = data['value_temp'][idx : idx+size_of_sample]
        d2 = data['value_hum'][idx : idx+size_of_sample]
        d3 = data['value_acid'][idx : idx+size_of_sample]
        # add them in a manner [temperature, humidity, acidity] (to the sample)
        sample = np.concatenate((d1,d2,d3))
        samples.append(sample)

    return np.array(samples)

# created for use with LSTM
def create_basic_data2(data: dict):
    size_of_sample = 30
    number_of_samples = len(data['value_temp']) - size_of_sample + 1
    samples = []
    for i in range(number_of_samples):
        sample = np.column_stack((
            data['value_temp'][i:i + size_of_sample],
            data['value_hum'][i:i + size_of_sample],
            data['value_acid'][i:i + size_of_sample]
        ))
        samples.append(sample)
    return np.array(samples)


def filter_savgol(data: dict) -> dict:
    smoothed1 = savgol_filter(data['value_temp'], window_length=10, polyorder=2)
    smoothed2 = savgol_filter(data['value_hum'], window_length=10, polyorder=2)
    smoothed3 = savgol_filter(data['value_acid'], window_length=10, polyorder=2)
    #smoothed4 = savgol_filter(data['value_PV'], window_length=10, polyorder=2)
    data['value_temp'] = smoothed1
    data['value_hum'] = smoothed2
    data['value_acid'] = smoothed3
    #data['value_PV'] = smoothed4
    return data


def filter_exponentialsmoothing(data: dict) -> dict:
    model1 = ExponentialSmoothing(data['value_temp'], seasonal_periods=12, trend='add', seasonal='add').fit()
    model2 = ExponentialSmoothing(data['value_hum'], seasonal_periods=12, trend='add', seasonal='add').fit()
    model3 = ExponentialSmoothing(data['value_acid'], seasonal_periods=12, trend='add', seasonal='add').fit()
    #model4 = ExponentialSmoothing(data['value_PV'], seasonal_periods=12, trend='add', seasonal='add').fit()
    smoothed1 = model1.predict(start=len(data['value_temp']), end=len(data['value_temp'])+11)
    smoothed2 = model2.predict(start=len(data['value_temp']), end=len(data['value_temp'])+11)
    smoothed3 = model3.predict(start=len(data['value_temp']), end=len(data['value_temp'])+11)
    #smoothed4 = model4.predict(start=len(data['value_temp']), end=len(data['value_temp'])+11)
    data['value_temp'] = smoothed1
    data['value_hum'] = smoothed2
    data['value_acid'] = smoothed3
    #data['value_PV'] = smoothed4
    return data


def filter_lowess(data: dict) -> dict:
    smoothed1 = np.array(lowess(data['value_temp'], range(len(data['value_temp'])), frac=0.1))[:,1]
    smoothed2 = np.array(lowess(data['value_hum'], range(len(data['value_hum'])), frac=0.1))[:,1]
    smoothed3 = np.array(lowess(data['value_acid'], range(len(data['value_acid'])), frac=0.1))[:,1]
    smoothed4 = np.array(lowess(data['value_PV'], range(len(data['value_PV'])), frac=0.1))[:,1]
    data['value_temp'] = smoothed1
    data['value_hum'] = smoothed2
    data['value_acid'] = smoothed3
    data['value_PV'] = smoothed4
    return data


def filter_kalman(data: dict) -> dict:
    kf1 = KalmanFilter(transition_matrices=[1],
                      observation_matrices=[1],
                      initial_state_mean=data['value_temp'].iloc[0],
                      initial_state_covariance=1,
                      observation_covariance=1,
                      transition_covariance=0.01)
    kf2 = KalmanFilter(transition_matrices=[1],
                       observation_matrices=[1],
                       initial_state_mean=data['value_hum'].iloc[0],
                       initial_state_covariance=1,
                       observation_covariance=1,
                       transition_covariance=0.01)
    kf3 = KalmanFilter(transition_matrices=[1],
                       observation_matrices=[1],
                       initial_state_mean=data['value_acid'].iloc[0],
                       initial_state_covariance=1,
                       observation_covariance=1,
                       transition_covariance=0.01)
    kf4 = KalmanFilter(transition_matrices=[1],
                       observation_matrices=[1],
                       initial_state_mean=data['value_PV'].iloc[0],
                       initial_state_covariance=1,
                       observation_covariance=1,
                       transition_covariance=0.01)
    state_means1, _ = kf1.filter(data['value_temp'])
    state_means1 = state_means1.flatten()

    state_means2, _ = kf2.filter(data['value_hum'])
    state_means2 = state_means2.flatten()

    state_means3, _ = kf3.filter(data['value_acid'])
    state_means3 = state_means3.flatten()

    state_means4, _ = kf4.filter(data['value_PV'])
    state_means4 = state_means4.flatten()

    data['value_temp'] = state_means1
    data['value_hum'] = state_means2
    data['value_acid'] = state_means3
    data['value_PV'] = state_means4
    return data


# just a skeleton i could be working on
def preprocessing(samples: np.array):

    # convert the NumPy array to a tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices(samples)

    def batch_format_fn(element):
        """Flatten a batch and return the features as OrderedDict"""
        return collections.OrderedDict(x=tf.reshape(element['data'], [-1, 90]),
                                        y=tf.reshape(element['label'], [-1, 1]))
    return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER, seed=1).batch(
            BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)


# current preprocessing (for lstm)
def preprocess():
    # take all the names in the directory
    data_dir = "/mnt/d/user/PycharmProjects/Platform-for-prototyping-fl-in-IoT/dane"
    file_names = os.listdir(data_dir)
    data = [pd.read_csv(os.path.join(data_dir, file)) for file in file_names]

    # right now df_RuralloT_002.csv
    oneFileDict = data[1].to_dict(orient='list')

    # normalise the data
    normalized_data = normalize(deepcopy(oneFileDict), deepcopy(oneFileDict))[0]

    # create the samples
    samples = create_basic_data2(normalized_data)
    return samples

    # samples = create_basic_data((normalise(copy(oneFileDict), copy(oneFileDict)))[0])
    #
    # predictions = outlier_detector(samples)
    #
    # preprocessed_dataset = preprocessing(samples)
    #
    # sample_batch = tf.nest.map_structure(lambda x: x.numpy(),
    #                                      next(iter(preprocessed_dataset)))
    # #print(sample_batch)
    # for batch in preprocessed_dataset.take(1):
    #     print(batch.shape)

# testing code for a machine, if it works, tensorflow works on your machine
@tff.federated_computation
def hello_world():
    return 'Hello, World!'



#emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

def test_preprocess(dataset):
    def batch_format_fn(element):
        return (tf.reshape(element['pixels'], [-1, 28, 28, 1]), tf.reshape(element['label'], [-1, 1]))
    return dataset.repeat(10).batch(BATCH_SIZE).map(batch_format_fn)



if __name__ == "__main__":
    # Load the preprocessed samples
    samples = preprocess()  # This should return a 3D array like `(5631, 30, 3)`

    # Configure numpy print options for better output visualization
    np.set_printoptions(threshold=np.inf, suppress=True, precision=2)

    # Load CSV files to find min/max values
    data_dir = "/mnt/d/user/PycharmProjects/Platform-for-prototyping-fl-in-IoT/dane"
    file_names = os.listdir(data_dir)
    data = [pd.read_csv(os.path.join(data_dir, file)) for file in file_names]

    print(file_names[1])

    # Extract min/max values from the relevant file
    oneFileDict = data[1].to_dict(orient='list')
    min_values = {
        'value_temp': min(oneFileDict['value_temp']),
        'value_hum': min(oneFileDict['value_hum']),
        'value_acid': min(oneFileDict['value_acid'])
    }
    max_values = {
        'value_temp': max(oneFileDict['value_temp']),
        'value_hum': max(oneFileDict['value_hum']),
        'value_acid': max(oneFileDict['value_acid'])
    }

    print("Min Values:", min_values)
    print("Max Values:", max_values)

    # Verify the shape of the samples array to confirm data dimensions
    print("Shape of samples:", samples.shape)  # Expected shape: `(5631, 30, 3)`

    # Define the LSTM model
    model = Sequential([
        LSTM(50, input_shape=(30, 3), return_sequences=False),  # Predict the next single row
        Dense(3)  # Predicts 3 features (columns)
    ])

    model.compile(optimizer="adam", loss="mean_squared_error")

    # Use the last 30 samples as input
    x = samples[-30:, :, :]  # Shape: `(30, 30, 3)`

    # The target is a single row following the last window
    y = samples[-1, -1, :].reshape(1, 3)  # Shape: `(1, 3)`

    # Ensure input and target shapes are printed for debugging
    print(x.shape)  # Expected: `(30, 30, 3)`
    print(y.shape)  # Expected: `(1, 3)`

    # Train the model on the last 30 windows with repeated targets
    model.fit(x, np.tile(y, (30, 1)), epochs=20, batch_size=5, validation_split=0.0)

    # Predict the next row after the last window
    prediction = model.predict(x[-1].reshape(1, 30, 3))  # Use the last window to predict the next row

    denormalized_prediction = denormalize(prediction, min_values, max_values)

    # Print the predicted results
    print("Predicted next value:", prediction)
    print("Predicted next value (denormalized):", denormalized_prediction)