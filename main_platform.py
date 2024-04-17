import time
from copy import copy
from datetime import datetime, timedelta
import os

import numpy as np
import tensorflow as tf
import csv
import pandas as pd

import matplotlib.pyplot as plt

from GUI.custom_architecture_widget import ArchitectureWidget
from GUI.generate_data_widget import GenerateDataWidget
from GUI.semisupervised_widget import SemiSupervisedWidget
from logic.BetterDataClass import BetterDataClass
from logic.Outlier_detectors import outlier_detector
from logic.dataProcesing import find_interrupts_withTime, find_shift_in_timeseries, normalise, filter_savgol, \
    filter_lowess, filter_exponentialsmoothing, create_basic_data
from logic.wrappers import time_wrapper

plt.style.use('dark_background')
# from tensorflow.keras.applications import ResNet50 <- sprawic by bylo zainstalowane tam sa modele
from PySide6.QtGui import Qt, QStandardItemModel

from PySide6.QtWidgets import QMainWindow, QHBoxLayout, QVBoxLayout, QWidget, QLabel, QComboBox
import keras

from GUI.button_widget import ButtonMenuHandler
from GUI.parameters_widget import ParametersHandler
from keras.applications import ResNet50
from colorama import Fore



# a class that defines how a GUI window will be created
class PlatformWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Platform for prototyping federated learning in IoT")
        #  50,50 = start position
        #  1600, 800 = dimensions of the window
        self.setGeometry(50, 50, 1600, 800)
        self.set_layout()
        with open(f"{os.getcwd()}//GUI//stylesheets//background.stylesheet") as file:
            self.setStyleSheet(file.read())
        # self.setStyleSheet("background-color: #355C7D;")
        self.oneFileDict = None
        self.good = None
        self.samples = None
        self.sample_id = 0
        self.predictions = None




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
        Reads data from given directory into pd dataframe and then converts to dict
        for test reasons only 1 file is used
        :return:
        """
        # take all the names in the directory
        file_names = os.listdir(".//dane")
        self.data = [pd.read_csv(f'.//dane//{file}') for file in file_names]
        self.BetterData = [BetterDataClass(data) for data in self.data]
        print("brek")
        # right now df_RuralloT_002.csv
        self.oneFileDict = self.data[1].to_dict(orient='list')
        print("brek")
        self.samples = create_basic_data((normalise(copy(self.oneFileDict), copy(self.oneFileDict)))[0])
        # self.samples = create_basic_data(self.oneFileDict)
        self.predictions = outlier_detector(self.samples)
        print("bre")
        for data in self.data:
            print(type(data))
        # self.test_widget = GenerateDataWidget(self.data)
        # self.test_widget.show()
        for data in self.data:
            print(type(data))

        self.test_widget2 = SemiSupervisedWidget(self.oneFileDict, self.BetterData)
        self.test_widget2.show()


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
    def find_empty2(self, data1, data2) -> None:

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
        fig, axs = plt.subplots(4,4)
        fig.suptitle('data shift noticed')

        for i in range(len(axs)):
            for j in range(len(axs[i])):
                axs[i,j].grid(True)
        if self.good is not None:
            for i, good in enumerate(self.good):

                if i == 6:

                    t = np.arange(0, len(good["time"]))
                    axs[0,0].plot(t, good["value_temp"], c='b', label="temperature")
                    axs[1,0].plot(t, good["value_hum"], c='g', label="humidity")
                    axs[2,0].plot(t, good["value_acid"], c='y', label="acid")
                    axs[3,0].plot(t, good["value_PV"], c='r', label="PV")
                    # good = filter_savgol(good)
                    # self.good[6] = filter_savgol(good)
                    good = filter_lowess(good)
                    self.good[6] =good
                    axs[0, 2].plot(t, good["value_temp"], c='b', label="temperature")
                    axs[1, 2].plot(t, good["value_hum"], c='g', label="humidity")
                    axs[2, 2].plot(t, good["value_acid"], c='y', label="acid")
                    axs[3, 2].plot(t, good["value_PV"], c='r', label="PV")

                    print(good["time"])
                if i == 7:
                    print(good["time"])
                    t = np.arange(0, len(good["time"]))
                    axs[0, 1].plot(t, good["value_temp"], c='b', label="temperature")
                    axs[0, 1].legend()
                    axs[1, 1].plot(t, good["value_hum"], c='g', label="humidity")
                    axs[1, 1].legend()
                    axs[2, 1].plot(t, good["value_acid"], c='y', label="acid")
                    axs[2, 1].legend()
                    axs[3, 1].plot(t, good["value_PV"], c='r', label="PV")
                    axs[3, 1].legend()

                    # good = filter_savgol(good)
                    # self.good[7] = filter_savgol(good)
                    good = filter_lowess(good)
                    self.good[7] = good
                    axs[0, 3].plot(t, good["value_temp"], c='b', label="temperature")
                    axs[1, 3].plot(t, good["value_hum"], c='g', label="humidity")
                    axs[2, 3].plot(t, good["value_acid"], c='y', label="acid")
                    axs[3, 3].plot(t, good["value_PV"], c='r', label="PV")
            plt.show(block=True)

    def connect_timeseries(self):
        idx1, idx2 = 6, 7
        if self.good is not None:
            n_missing = find_shift_in_timeseries(self.good[idx1], self.good[idx2])
            connected_length = n_missing + len(self.good[idx1]) + len(self.good[idx2])
            normalised1, normalised2 = normalise(self.good[idx1], self.good[idx2])
            # fig, axs = plt.subplots(4, 1)
            # fig.suptitle('data shift fixed')
            # plt.legend()
            fig, ax = plt.subplots()
            colors = [["#0000ff", "#00ccff"],
                      ["#ff00ff", "#660066"],
                      ["#ff6666", "#cc0000"],
                      ["#66ff66", "#009933"],
            ]
            for idx, key in enumerate(["value_temp","value_hum","value_acid","value_PV"]):
                data = pd.Series(normalised1[key]+[np.nan for _ in range(n_missing)]+normalised2[key])
                data = data.interpolate()
                data.name= key
                # data.interpolate(method='polynomial', order=2)

                print(data[100])
                print(n_missing)
                part1 = data[:len(normalised1[key])]
                part2 = data[len(normalised1[key]):len(normalised1[key])+n_missing]
                part3 = data[(len(normalised1[key])+n_missing):]
                part1.plot(ax=ax,title=f"connected 2 timeseries with naive interpolation", legend=True, color =colors[idx][0])
                part2.plot(ax=ax, color =colors[idx][1])
                part3.plot(ax=ax, color =colors[idx][0])
                # plot.savefig(f'{key}.pdf')




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

    def outlier_check(self):
        if self.samples is not None and self.predictions is not None:
            fig, axs = plt.subplots(4, 2)
            sample = self.samples[self.sample_id]
            prediction = self.predictions[self.sample_id]

            fig.suptitle(f'show outliers id:{self.sample_id} outlier?:{"True" if prediction==-1 else "False"}')
            t = np.arange(0, 20)
            axs[0, 0].plot(t, sample[:20], c='b', label="temperature")
            axs[0, 0].legend()
            axs[1, 0].plot(t, sample[20:40], c='g', label="humidity")
            axs[1, 0].legend()
            axs[2, 0].plot(t, sample[40:60], c='y', label="acid")
            axs[2, 0].legend()
            axs[3, 0].plot(t, sample[60:], c='r', label="PV")
            axs[3, 0].legend()

            plt.show(block=True)
            self.sample_id += 1
    def better_outlier_check(self):

        if self.samples is not None and self.predictions is not None:
            stretch = 30
            spans = []
            start = 0
            stop = stretch
            print(len(self.samples) + stretch)
            while(stop < len(self.samples) + stretch):

                print(f"start while {start}:{stop}")


                print(self.predictions[start])
                if self.predictions[start]==-1:
                    for i in range(start,len(self.predictions),1):
                        if -1 in self.predictions[stop-stretch:stop]:
                            stop += 1
                        else:
                            span = [start, stop, False]
                            spans.append(span)
                            start = stop
                            stop += stretch
                            break
                else:
                    if -1 in self.predictions[stop-stretch:stop]:
                        for z, n in enumerate(self.predictions[stop-stretch:stop]):
                            if n == -1:
                                stop = start+z
                                print("    ", start , stop, z)
                                span = [start, stop, True]
                                spans.append(span)
                                start = stop
                                stop += stretch

                                break
                    else:
                        for i in range(start, len(self.predictions), 1):
                            if -1 in self.predictions[stop - stretch:stop]:
                                span = [start, stop, True]
                                spans.append(span)
                                start = stop
                                stop += stretch
                                break
                            else:
                                stop += 1

            fig, axs = plt.subplots(4, 1)
            colors = [["#00ccff", "#0000ff"],
                      ["#ff00ff", "#660066"],
                      ["#ff6666", "#cc0000"],
                      ["#66ff66", "#009933"]]
            for k, span in enumerate(spans):
                if k==0:
                    print(span[2])
                t = np.arange(span[0], span[1]+1)
                if len(spans)-1==k:
                    axs[0].plot(t, self.oneFileDict["value_temp"][span[0]: 1 + span[1]],
                                c=colors[0][0] if span[2] else colors[0][1], label="temperature")

                    axs[1].plot(t, self.oneFileDict["value_hum"][span[0]: 1 + span[1]],
                                c=colors[1][0] if span[2] else colors[1][1], label="humidity")

                    axs[2].plot(t, self.oneFileDict["value_acid"][span[0]: 1 + span[1]],
                                c=colors[2][0] if span[2] else colors[2][1], label="acid")

                    axs[3].plot(t, self.oneFileDict["value_PV"][span[0]: 1 + span[1]],
                                c=colors[3][0] if span[2] else colors[3][1], label="PV")
                else:
                    axs[0].plot(t, self.oneFileDict["value_temp"][span[0] : 1+span[1]], c=colors[0][0] if span[2] else colors[0][1])

                    axs[1].plot(t, self.oneFileDict["value_hum"][span[0] : 1+span[1]], c=colors[1][0] if span[2] else colors[1][1])

                    axs[2].plot(t, self.oneFileDict["value_acid"][span[0] : 1+span[1]], c=colors[2][0] if span[2] else colors[2][1])

                    axs[3].plot(t, self.oneFileDict["value_PV"][span[0] : 1+span[1]], c=colors[3][0] if span[2] else colors[3][1])
            z = 0

            for idxx, good in enumerate(self.good):
                if idxx >= len(self.good)-2:
                    pass
                else:
                    print(len(good["value_temp"]))
                    z+= len(good["value_temp"])
                    axs[0].plot(z,self.oneFileDict["value_temp"][z], c="y" , marker = "o")
                    axs[1].plot(z,self.oneFileDict["value_hum"][z], c="y" , marker = "o")
                    axs[2].plot(z,self.oneFileDict["value_acid"][z], c="y" , marker = "o")
                    axs[3].plot(z,self.oneFileDict["value_PV"][z], c="y" , marker = "o")


            axs[0].legend()
            axs[0].grid(True)
            axs[1].legend()
            axs[1].grid(True)
            axs[2].legend()
            axs[2].grid(True)
            axs[3].legend()
            axs[3].grid(True)
            plt.show()
            print("b")



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

        self.architecture = ArchitectureWidget()

        self.modelsCombo.setMinimumHeight(70)
        self.modelsCombo.setMaximumWidth(250)
        self.modelsCombo.setStyleSheet("background-color: #518dbf; font-size: 20px; border-radius 10px; border: 3px solid #0033cc;")
        self.modelsCombo.addItem("Model Test")
        self.modelsCombo.addItem("Model Test2")
        self.horizontalLayout1.addWidget(self.label1)
        self.horizontalLayout2.addWidget(self.architecture)
        self.horizontalLayout1.addWidget(self.modelsCombo)

        self.pushButtonMenu.customButton1.clicked.connect(self.load_model)
        self.pushButtonMenu.customButton2.clicked.connect(self.read_data)
        self.pushButtonMenu.customButton3.clicked.connect(self.plot_data)
        self.pushButtonMenu.customButton4.clicked.connect(self.find_good)
        self.pushButtonMenu.customButton5.clicked.connect(self.plot_data_good2)
        self.pushButtonMenu.customButton6.clicked.connect(self.find_empty)
        self.pushButtonMenu.customButton7.clicked.connect(self.connect_timeseries)
        self.pushButtonMenu.customButton10.clicked.connect(self.outlier_check)
        self.pushButtonMenu.customButton11.clicked.connect(self.better_outlier_check)




