import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from statsmodels.tsa.seasonal import seasonal_decompose


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

        self.button1 = QPushButton("start")
        self.button1.clicked.connect(self.decomposition)
        self.mainLayout.addWidget(self.button1)

        single = self.get_single_data(self.data, 1)
        points = self.extractor(single)



        avg = self.movingAverage2(points=points[0])

        # avg = avg[50:]
        # avg = avg[:len(avg)-50]
        # avg = np.array(avg)
        # p = np.array(points[0])
        # p = p[50:]
        # p = p[:len(p) - 50]
        #
        # peaks, _ = find_peaks(avg)
        # plt.plot(peaks, avg[peaks], "x")
        # plt.plot(np.arange(len(avg)), avg)
        # plt.plot(np.arange(len(avg)), p)
        # plt.show()


        self.data[1]['time'] = pd.to_datetime(self.data[1]['time'])
        self.data[1].set_index('time', inplace=True)
        self.data[1] = self.data[1].iloc[50:-50]
        print(self.data[1])
        self.seasonal(self.data[1])

    def get_single_data(self, data, id:int=0):
        single_IoT_sensor_data = data[id].to_dict(orient='list')
        return single_IoT_sensor_data


    def extractor(self, data):
        d1 = data['value_temp']
        d2 = data['value_hum']
        d3 = data['value_acid']
        d4 = data['value_PV']
        return d1, d2, d3, d4

    def seasonal(self, points):
        decomposition = seasonal_decompose(points['value_temp'], model='additive', period=24)
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid
        # residual.plot()
        # seasonal.plot()
        # trend.plot()
        # plt.show()
        plt.plot(points.index, points['value_temp'], label='Original Data')
        for i in range(10):
            self.naive(trend,seasonal,residual)
        # self.naive(trend,seasonal,residual)
        plt.show()


        # decomposition.plot()
        print("brek")



    def naive(self, trend, seasonal, residual):
        # 1. Modify the trend (for example, apply a slight linear increase)
        trend_mod = trend + np.linspace(0, 1, len(trend))

        # 2. Replicate or modify the seasonal component
        seasonal_mod = seasonal  # Assuming the seasonality will remain constant

        # 3. Randomize the residuals by shuffling
        residual_mod = np.random.permutation(residual.dropna())  # Drop NaNs before shuffling

        # 4. Make sure the length of the modified residual component matches the others
        residual_mod = np.concatenate((residual_mod, residual_mod[:len(trend) - len(residual_mod)]))

        # Recombine the components to create synthetic data
        synthetic_data = trend_mod + seasonal_mod + residual_mod

        # Ensure the synthetic data matches the length of the original components
        synthetic_data = synthetic_data[:len(trend)]
        synthetic_data.plot()

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
