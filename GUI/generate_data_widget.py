import random

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

from GUI.image_widget import ImageWidget
from logic.dataProcesing import find_interrupts_withTime


def find_avg_time(dataframe):
    time_diffs = dataframe.index.to_series().diff()

    # Convert the differences to minutes
    time_diffs_in_minutes = time_diffs.dt.total_seconds() / 60

    # Calculate the average of these differences in minutes
    average_time_diff_minutes = time_diffs_in_minutes.mean()
    return average_time_diff_minutes


def extractor(dataframe, key):
    dictionary_data = dataframe.to_dict(orient='list')
    return dictionary_data[key]


def decompose(dataframe, key, debug=None):
    avg_time =find_avg_time(dataframe)
    points = extractor(dataframe, key=key)

    print(points)
    print(f"avg_time = {avg_time} len_points = {len(points)}")
    decomposition = seasonal_decompose(points, model='additive', period=int(1440//avg_time))
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    if debug is not None:
        decomposition.plot()
        plt.show()
    return trend, seasonal, residual

def movingAverage2(points, window_size = 30):
    h = window_size//2
    moving_averages = []
    for i in range(len(points)):
        if i < h:
            current_window = points[0:i + h]
            window_average = sum(current_window) / window_size
            moving_averages.append(window_average)
        elif i + h > len(points):
            current_window = points[i-h+1:len(points)]
            window_average = sum(current_window) / window_size
            moving_averages.append(window_average)
        else:
            current_window = points[i-h+1:i+h+1]
            window_average = sum(current_window) / window_size
            moving_averages.append(window_average)
    return moving_averages


def naive(trend, seasonals, residuals):
    seasonals = [np.where(np.isnan(s), 0, s) for s in seasonals]
    residuals = [np.where(np.isnan(r), 0, r) for r in residuals]
    trend_mod = trend + np.linspace(0, 1, len(trend))
    seasonal_mod = np.zeros_like(trend)
    residual_mod = np.zeros_like(trend)

    segment_size = len(trend) // 10
    i = 0


    while(i < len(trend)):
        min_step = 100
        seasonal = random.choice(seasonals)
        residual = random.choice(residuals)
        start1 = 100000000
        while(start1>len(seasonal)-min_step):
            start1 = random.randint(0, len(seasonal))
        end1=0
        while (end1 < min_step):
            end1 = random.randint(start1, len(seasonal))
        diff=end1-start1
        if i+diff > len(trend):
            left = len(trend)-i
            diff= left
            end1 = start1+left
        seasonal_mod[i:i+diff] = seasonal[start1:end1]

        start2 = 100000000
        while (start2 > len(residual) - min_step):
            start2 = random.randint(0, len(residual))
        end2 = 0
        while (end2 < min_step):
            end2 = random.randint(start2, len(residual))
        diff = end2 - start2
        if i+diff > len(trend):
            left = len(trend)-i
            diff= left
            end2 = start2+left
        residual_mod[i:i + diff] = residual[start2:end2]
        i +=diff



    synthetic_data = trend_mod + seasonal_mod + residual_mod
    return synthetic_data


class GenerateDataWidget(QWidget):
    def __init__(self, data):
        super(GenerateDataWidget, self).__init__()
        self.setMinimumHeight(400)
        self.setMinimumWidth(400)
        self.setStyleSheet("background-color: #F67280;")
        self.mainLayout = QVBoxLayout()
        self.mainLayout.setAlignment(Qt.AlignTop)
        self.setLayout(self.mainLayout)
        self.data = data


        # self.imagewidget = ImageWidget()
        # self.imagewidget.show()


        self.button1 = QPushButton("start")
        self.button1.clicked.connect(self.decomposition)
        self.mainLayout.addWidget(self.button1)


        # self.generation()



        # single = self.get_single_data(self.data, 1)
        # points = self.extractor(single)
        #
        # self.good = find_interrupts_withTime(single)
        # print(self.good)
        # print(len(self.good))
        # # self.good = [good for good in self.good if len(good['value_temp']) >1000]
        #
        # avg = self.movingAverage2(points=points[0])
        # df = pd.DataFrame(self.good)
        # print(df)
        # print(df.iloc[0])
        # print(df.iloc[-1])
        # df['time'] = pd.to_datetime(df['time'])
        # df.set_index('time', inplace=True)
        # df = df.iloc[50:-50]
        #
        #
        # # self.seasonal(df)
        #
        # self.data[1]['time'] = pd.to_datetime(self.data[1]['time'])
        # self.data[1].set_index('time', inplace=True)
        # self.data[1] = self.data[1].iloc[50:-50]
        # time_diffs = self.data[1].index.to_series().diff()
        #
        # # Convert the differences to minutes
        # time_diffs_in_minutes = time_diffs.dt.total_seconds() / 60
        #
        # # Calculate the average of these differences in minutes
        # average_time_diff_minutes = time_diffs_in_minutes.mean()
        #
        # print(f"Average time interval in minutes: {average_time_diff_minutes}")
        # self.seasonal(self.data[1])

    def generation(self):
        id = 1
        single = self.get_single_data(self.data, id)

        main_ts = self.data[id]
        main_ts['time'] = pd.to_datetime(main_ts['time'])
        main_ts.set_index('time', inplace=True)
        main_ts = main_ts.iloc[50:-50]

        self.good = find_interrupts_withTime(single)


        self.good = [pd.DataFrame(good) for good in self.good if len(good['value_temp']) > 1000]
        print(f"left {len(self.good)}")
        for good in self.good:
            good['time'] = pd.to_datetime(good['time'])
            good.set_index('time', inplace=True)

        keys = ["value_temp","value_hum","value_acid"]

        tsr0 = decompose(self.good[0],key=keys[0],debug=True)
        tsr1 = decompose(self.good[1],key=keys[0],debug=True)
        main_tsr = decompose(main_ts,key=keys[0],debug=True)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(5):
            synthetic = naive(main_tsr[0],[tsr0[1],tsr1[1]],[tsr0[2],tsr1[2]])
            synthetic = synthetic+i
            print(f"len synthetic = {len(synthetic)}")
            ax.plot(synthetic)
        ax.plot(single[keys[0]])

        plt.show()

    def get_single_data(self, data, id:int=0):
        single_IoT_sensor_data = data[id].copy().to_dict(orient='list')
        return single_IoT_sensor_data


    def extractor(self, data):
        d1 = data['value_temp']
        d2 = data['value_hum']
        d3 = data['value_acid']
        d4 = data['value_PV']
        return d1, d2, d3, d4

    def seasonal(self, points):
        physical = "value_temp"

        decomposition = seasonal_decompose(points[physical], model='additive', period=143)
        decomposition2 = seasonal_decompose(points[physical], model='additive', period=48)
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid
        # residual.plot()
        # seasonal.plot()
        # trend.plot()
        decomposition.plot()

        # plt.show()
        # plt.plot(points.index, points[physical], label='Original Data')
        for i in range(1):
            self.naive(trend,seasonal,residual)
            # self.naive(decomposition2.trend,decomposition2.seasonal,decomposition2.resid)
        # self.naive(trend,seasonal,residual)
        plt.show()





    def naive(self, trend, seasonal, residual):
        # 1. Modify the trend (for example, apply a slight linear increase)
        trend_mod = trend + np.linspace(0, 1, len(trend))

        # # 2. Replicate or modify the seasonal component
        # seasonal_mod = seasonal  # Assuming the seasonality will remain constant

        # 2. Replicate or modify the seasonal component with slight shuffling
        # Define the size of each segment to shuffle
        segment_size = len(seasonal) // 20  # For example, divide the seasonal component into 10 segments
        seasonal_mod = np.power(seasonal.copy(),3)  # Make a copy to avoid modifying the original
        for i in range(0, len(seasonal), segment_size):
            segment_end = i + segment_size
            if segment_end > len(seasonal):
                segment_end = len(seasonal)
            shuffled_segment = np.random.permutation(seasonal_mod[i:segment_end])
            seasonal_mod[i:segment_end] = shuffled_segment

        # # 3. Randomize the residuals by shuffling
        # residual_mod = np.random.permutation(residual.dropna())  # Drop NaNs before shuffling
        #
        # # 4. Make sure the length of the modified residual component matches the others
        # residual_mod = np.concatenate((residual_mod, residual_mod[:len(trend) - len(residual_mod)]))

        positive_residuals = np.sqrt(residual[residual >= 0])
        negative_residuals = -np.sqrt(-residual[residual < 0])

        # Combine the transformed residuals
        residual_mod = np.concatenate((positive_residuals, negative_residuals))

        # Randomize the transformed residuals by shuffling
        np.random.shuffle(residual_mod)

        # Ensure the modified residual component matches the length of the others
        residual_mod = np.concatenate((residual_mod, residual_mod[:len(trend) - len(residual_mod)]))

        # Recombine the components to create synthetic data
        synthetic_data = trend_mod + seasonal_mod + residual_mod

        # Ensure the synthetic data matches the length of the original components
        synthetic_data = synthetic_data[:len(trend)]
        # synthetic_data.plot()

    # def movingAverage(self, points, window_size = 30):
    #
    #     moving_averages = []
    #     for i in range(len(points)):
    #         if i < window_size:
    #             current_window = points[0:i + 1]
    #             window_average = sum(current_window) / window_size
    #             moving_averages.append(window_average)
    #         else:
    #             current_window = points[i-window_size+1:i+1]
    #             window_average = sum(current_window) / window_size
    #             moving_averages.append(window_average)
    #     return moving_averages

    def movingAverage2(self, points, window_size = 30):
        h = window_size//2
        moving_averages = []
        for i in range(len(points)):
            if i < h:
                current_window = points[0:i + h]
                window_average = sum(current_window) / window_size
                moving_averages.append(window_average)
            elif i + h > len(points):
                current_window = points[i-h+1:len(points)]
                window_average = sum(current_window) / window_size
                moving_averages.append(window_average)
            else:
                current_window = points[i-h+1:i+h+1]
                window_average = sum(current_window) / window_size
                moving_averages.append(window_average)
        return moving_averages

    def decomposition(self):
        pass
        self.imagewidget.update_image()
