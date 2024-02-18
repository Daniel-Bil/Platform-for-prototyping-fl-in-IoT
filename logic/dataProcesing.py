import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from logic.wrappers import time_wrapper
from scipy.signal import savgol_filter
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.nonparametric.smoothers_lowess import lowess
from pykalman import KalmanFilter

@time_wrapper
def find_interrupts_withPV(data: dict | pd.DataFrame):



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
def find_interrupts_withTime(data: dict | pd.DataFrame):
    '''
    Function that finds shift in time series and divides into list of good timeseries
    :param data:
    :return: list
    '''
    print()
    number_of_measurements = len(data["time"])
    idxs = []
    for i in range(number_of_measurements-1):
        one = datetime.fromisoformat(data["time"][i].replace('Z', '+00:00'))
        two = datetime.fromisoformat(data["time"][i + 1].replace('Z', '+00:00'))
        if int((two - one).total_seconds()) / 60 > 30:
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
        start = idx + 1
    return good

@time_wrapper
def find_shift_in_timeseries(data1, data2):
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
def normalise(data1:dict, data2:dict):
    for key in ["value_temp","value_hum","value_acid","value_PV"]:

        max1 = max([max(data1[key]), max(data2[key])])
        min1 = min([min(data1[key]), min(data2[key])])
        data1[key] = [(d-min1)/(max1-min1) for d in data1[key]]
        data2[key] = [(d-min1)/(max1-min1) for d in data2[key]]
    return data1, data2


def filter_savgol(data):
    smoothed1 = savgol_filter(data['value_temp'], window_length=10, polyorder=2)
    smoothed2 = savgol_filter(data['value_hum'], window_length=10, polyorder=2)
    smoothed3 = savgol_filter(data['value_acid'], window_length=10, polyorder=2)
    smoothed4 = savgol_filter(data['value_PV'], window_length=10, polyorder=2)
    data['value_temp'] = smoothed1
    data['value_hum'] = smoothed2
    data['value_acid'] = smoothed3
    data['value_PV'] = smoothed4
    return data


def filter_exponentialsmoothing(data):
    model1 = ExponentialSmoothing(data['value_temp'], seasonal_periods=12, trend='add', seasonal='add').fit()
    model2 = ExponentialSmoothing(data['value_hum'], seasonal_periods=12, trend='add', seasonal='add').fit()
    model3 = ExponentialSmoothing(data['value_acid'], seasonal_periods=12, trend='add', seasonal='add').fit()
    model4 = ExponentialSmoothing(data['value_PV'], seasonal_periods=12, trend='add', seasonal='add').fit()
    smoothed1 = model1.predict(start=len(data['value_temp']), end=len(data['value_temp'])+11)
    smoothed2 = model2.predict(start=len(data['value_temp']), end=len(data['value_temp'])+11)
    smoothed3 = model3.predict(start=len(data['value_temp']), end=len(data['value_temp'])+11)
    smoothed4 = model4.predict(start=len(data['value_temp']), end=len(data['value_temp'])+11)
    data['value_temp'] = smoothed1
    data['value_hum'] = smoothed2
    data['value_acid'] = smoothed3
    data['value_PV'] = smoothed4
    return data


def filter_lowess(data):
    smoothed1 = lowess(data['value_temp'], range(len(data['value_temp'])), frac=0.1)
    smoothed2 = lowess(data['value_hum'], range(len(data['value_hum'])), frac=0.1)
    smoothed3 = lowess(data['value_acid'], range(len(data['value_acid'])), frac=0.1)
    smoothed4 = lowess(data['value_PV'], range(len(data['value_PV'])), frac=0.1)
    data['value_temp'] = smoothed1
    data['value_hum'] = smoothed2
    data['value_acid'] = smoothed3
    data['value_PV'] = smoothed4
    return data


def filter_kalman(data):
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


if __name__ == "__main__":
    x = {"value_PV": [1,2,3,4,5,56,6,8,7,78],
         "time": [1,2,3,4,5,56,6,8,7,78]}
    # find_interrupts(x)