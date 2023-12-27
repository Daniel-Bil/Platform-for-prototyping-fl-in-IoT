import time
from datetime import datetime, timedelta
import os

import numpy as np
import tensorflow as tf
import csv
import pandas as pd

import matplotlib.pyplot as plt

from logic.data_procesing import find_interrupts_withPV, find_interrupts_withTime, find_shift_in_timeseries, normalise
from logic.wrappers import time_wrapper

plt.style.use('dark_background')
# from tensorflow.keras.applications import ResNet50 <- sprawic by bylo zainstalowane tam sa modele
from PySide6.QtGui import Qt


from PySide6.QtWidgets import QMainWindow, QHBoxLayout, QVBoxLayout, QWidget, QLabel, QComboBox
import keras

from GUI.button_widget import ButtonMenuHandler
from GUI.parameters_widget import ParametersHandler
from keras.applications import ResNet50
from colorama import Fore

class PlatformWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Platform for prototyping federated learning in IoT")
        self.setGeometry(50, 50, 1600, 800)
        self.set_layout()
        self.setStyleSheet("background-color: #669999;")
        self.oneFileDict = None
        self.good = None

    @time_wrapper
    def load_model(self):
        """
        not used
        :return:
        """
        print(f"{self.modelsCombo.currentIndex()}  {self.modelsCombo.currentText()}")

    @time_wrapper
    def read_data(self):
        """
        Reads data from given file inot pd dataframe and inot dict
        :return:
        """
        print("load data")

        files = os.listdir(f"{os.getcwd()}//dane")
        time, value_temp, value_hum, value_acid, value_PV =[],[],[],[],[]
        df = pd.read_csv(f'{os.getcwd()}//dane//{files[0]}')
        print(df)
        with open(f"{os.getcwd()}//dane//{files[0]}") as file:
            reader = csv.DictReader(file)
            for line in reader:
                time.append(line["time"])
                value_temp.append(float(line["value_temp"]))
                value_hum.append(float(line["value_hum"]))
                value_acid.append(float(line["value_acid"]))
                value_PV.append(float(line["value_PV"]))
        self.oneFileDict = {"time": time,
                            "value_temp": value_temp,
                            "value_hum": value_hum,
                            "value_acid": value_acid,
                            "value_PV": value_PV}

    @time_wrapper
    def plot_data(self):
        """
        Plots whole data useless
        :return:
        """
        if self.oneFileDict is not None:
            t = np.arange(0, len(self.oneFileDict["time"]))
            plt.plot(t, self.oneFileDict["value_temp"], c='b', label="temperature")
            plt.plot(t, self.oneFileDict["value_hum"], c='g', label="humidity")
            plt.plot(t, self.oneFileDict["value_acid"], c='y', label="acid")
            plt.plot(t, self.oneFileDict["value_PV"], c='r', label="PV")
            plt.grid(True)
            plt.legend()
            plt.show()
        # for i in range(1000):
        #
        #     print(f"time:{self.oneFileDict['time'][i]} pvvalue:{self.oneFileDict['value_PV'][i]} dpv:{0 if i==0 or i==999 else self.oneFileDict['value_PV'][i]-self.oneFileDict['value_PV'][i+1]:^3.2f}")

    @time_wrapper
    def find_good(self):
        "finds good time series "
        self.good = find_interrupts_withTime(self.oneFileDict)
        for i, good in enumerate(self.good):
            v = self.check_time_differences(good["time"], timedelta(minutes=20))
            print(i, v)
            if not v:
               for j in range(len(good['time'])):
                   print(f"t:{good['time'][j]} pv:{good['value_PV'][j]} pvd:{0 if j==len(good['time'])-1 else good['value_PV'][j]-good['value_PV'][j+1]:^3.2f} ac:{good['value_acid'][j]} acd:{0 if j==len(good['time'])-1 else good['value_acid'][j]-good['value_acid'][j+1]:^3.2f}")
               print()

    @time_wrapper
    def find_empty(self):
        """
        Finds time difference between all time series and return info about number of missed points
        :return:
        """
        if self.good is not None:

            self.find_empty2(self.good[4],self.good[5])


            # for i in range(len(self.good)-1):
            #     avg1,avg2 = [],[]
            #     for j in range(len(self.good[i]["time"])-1):
            #         one = datetime.fromisoformat(self.good[i]["time"][j].replace('Z', '+00:00'))
            #         two = datetime.fromisoformat(self.good[i]["time"][j+1].replace('Z', '+00:00'))
            #         avg1.append(int((two-one).total_seconds())/60)
            #     number_of_missing_values = np.mean(avg1)
            #     start = datetime.fromisoformat(self.good[i]["time"][-1].replace('Z', '+00:00'))
            #     end = datetime.fromisoformat(self.good[i+1]["time"][0].replace('Z', '+00:00'))
            #     print(f"{Fore.GREEN}{start} {Fore.BLUE} {end} {Fore.RED} {end - start} {Fore.RESET}")
            #     difference_in_minutes = int((end - start).total_seconds() / 60)
            #     print(difference_in_minutes)
            #     print(f"number of missing values = {int(difference_in_minutes/number_of_missing_values)}")
            #     # print(start)
            #     # print(end)
            #     # print(end - start)

    @time_wrapper
    def find_empty2(self, data1, data2):
        avg1 = []
        for j in range(len(data1["time"]) - 1):
            print(data1["time"][j])
            print(data1["time"][j+1])
            print()
            one = datetime.fromisoformat(data1["time"][j].replace('Z', '+00:00'))
            two = datetime.fromisoformat(data1["time"][j + 1].replace('Z', '+00:00'))
            avg1.append(int((two - one).total_seconds()) / 60)

        number_of_missing_values = np.mean(avg1)

        print(avg1)
        start = datetime.fromisoformat(data1["time"][-1].replace('Z', '+00:00'))
        end = datetime.fromisoformat(data2["time"][0].replace('Z', '+00:00'))
        print(f"{Fore.GREEN}{start} {Fore.BLUE} {end} {Fore.RED} {end - start} {Fore.RESET}")
        difference_in_minutes = int((end - start).total_seconds() / 60)
        print(difference_in_minutes)
        print(f"number of missing values = {int(difference_in_minutes / number_of_missing_values)}")

    @time_wrapper
    def plot_data_good(self):
        """
        plots data *NEEDS FIX*
        :return:
        """
        if self.good is not None:
            for i,good in enumerate(self.good):

                if i == 4 or i == 5:
                    t = np.arange(0, len(good["time"]))
                    plt.plot(t, good["value_temp"], c='b', label="temperature")
                    plt.plot(t, good["value_hum"], c='g', label="humidity")
                    plt.plot(t, good["value_acid"], c='y', label="acid")
                    plt.plot(t, good["value_PV"], c='r', label="PV")


                    plt.grid(True)
                    plt.legend()
                    print(good["time"])
                    plt.show(block=True)

    @time_wrapper
    def plot_data_good2(self):
        """
        plots data *NEEDS FIX*
        :return:
        """
        fig, axs = plt.subplots(4,2)
        fig.suptitle('data shift noticed')
        plt.legend()
        for i in range(len(axs)):
            for j in range(len(axs[i])):
                axs[i,j].grid(True)
        if self.good is not None:
            for i, good in enumerate(self.good):

                if i == 4:
                    t = np.arange(0, len(good["time"]))
                    axs[0,0].plot(t, good["value_temp"], c='b', label="temperature")
                    axs[1,0].plot(t, good["value_hum"], c='g', label="humidity")
                    axs[2,0].plot(t, good["value_acid"], c='y', label="acid")
                    axs[3,0].plot(t, good["value_PV"], c='r', label="PV")

                    print(good["time"])
                if i == 5:
                    print(good["time"])
                    t = np.arange(0, len(good["time"]))
                    axs[0, 1].plot(t, good["value_temp"], c='b', label="temperature")
                    axs[1, 1].plot(t, good["value_hum"], c='g', label="humidity")
                    axs[2, 1].plot(t, good["value_acid"], c='y', label="acid")
                    axs[3, 1].plot(t, good["value_PV"], c='r', label="PV")

            plt.show(block=True)

    def connect_timeseries(self):
        if self.good is not None:
            n_missing = find_shift_in_timeseries(self.good[4], self.good[5])
            connected_length = n_missing + len(self.good[4]) + len(self.good[5])
            normalised1, normalised2 = normalise(self.good[4], self.good[5])
            # fig, axs = plt.subplots(4, 1)
            # fig.suptitle('data shift fixed')
            # plt.legend()
            for key in ["value_temp","value_hum","value_acid","value_PV"]:
                data = pd.Series(normalised1[key]+[np.nan for _ in range(n_missing)]+normalised2[key])
                data = data.interpolate()
                # data.interpolate(method='polynomial', order=2)

                print(data[100])
                plot = data.plot(title=f"{key}").get_figure()
                plot.savefig(f'{key}.pdf')




    @time_wrapper
    def check_time_differences(self, times, max_difference):
        """
        Informs whether time difference between 2 measurements is bigger than max_difference
        :param times:
        :param max_difference:
        :return:
        """
        # Convert ISO format strings to datetime objects
        datetime_objects = [datetime.fromisoformat(time.replace('Z', '+00:00')) for time in times]

        # Iterate over the datetime objects
        for i in range(len(datetime_objects) - 1):
            # Calculate the difference between adjacent times
            difference = datetime_objects[i + 1] - datetime_objects[i]
            # Check if the difference is greater than the allowed maximum
            if difference > max_difference:
                return False  # If any difference is too large, return False
        return True  # If all differences are within the limit, return True

    @time_wrapper
    def set_layout(self):

        self.mainLayout = QVBoxLayout()

        self.mainWidget = QWidget()
        self.mainWidget.setLayout(self.mainLayout)

        self.setCentralWidget(self.mainWidget)

        self.horizontalLayout1 = QHBoxLayout()
        self.horizontalLayout1.setAlignment(Qt.AlignTop)
        self.horizontalLayout2 = QHBoxLayout()
        # self.horizontalLayout2.setAlignment(Qt.AlignRight)

        self.horizontalLayout3 = QHBoxLayout()

        self.mainLayout.addLayout(self.horizontalLayout1)
        self.mainLayout.addLayout(self.horizontalLayout2)
        self.mainLayout.addLayout(self.horizontalLayout3)

        self.pushButtonMenu = ButtonMenuHandler()
        self.parametersMenu = ParametersHandler()
        self.horizontalLayout2.addWidget(self.parametersMenu)
        self.horizontalLayout2.addWidget(self.pushButtonMenu)



        self.verticalLayout1 = QVBoxLayout()
        self.verticalLayout2 = QVBoxLayout()

        self.label1 = QLabel("models: ")
        self.modelsCombo = QComboBox()
        self.modelsCombo.setMinimumHeight(70)
        self.modelsCombo.setMaximumWidth(250)
        self.modelsCombo.setStyleSheet("background-color: DodgerBlue; font-size: 20px; border-radius 10px; border: 3px solid #0033cc;")
        self.modelsCombo.addItem("Model Test")
        self.modelsCombo.addItem("Model Test2")
        self.horizontalLayout1.addWidget(self.label1)
        self.horizontalLayout1.addWidget(self.modelsCombo)

        self.pushButtonMenu.customButton1.clicked.connect(self.load_model)
        self.pushButtonMenu.customButton2.clicked.connect(self.read_data)
        self.pushButtonMenu.customButton3.clicked.connect(self.plot_data)
        self.pushButtonMenu.customButton4.clicked.connect(self.find_good)
        self.pushButtonMenu.customButton5.clicked.connect(self.plot_data_good2)
        self.pushButtonMenu.customButton6.clicked.connect(self.find_empty)
        self.pushButtonMenu.customButton7.clicked.connect(self.connect_timeseries)




