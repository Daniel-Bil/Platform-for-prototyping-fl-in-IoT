import os

import numpy as np
import tensorflow as tf
import csv
import matplotlib.pyplot as plt
plt.style.use('dark_background')
# from tensorflow.keras.applications import ResNet50 <- sprawic by bylo zainstalowane tam sa modele
from PySide6.QtGui import Qt

from PySide6.QtWidgets import QMainWindow, QHBoxLayout, QVBoxLayout, QWidget, QLabel, QComboBox
import keras

from GUI.button_widget import ButtonMenuHandler
from GUI.parameters_widget import ParametersHandler
from keras.applications import ResNet50


class PlatformWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Platform for prototyping federated learning in IoT")
        self.setGeometry(50, 50, 1600, 800)
        self.set_layout()
        self.setStyleSheet("background-color: #669999;")
        self.oneFileDict = None

    def load_model(self):
        print(f"{self.modelsCombo.currentIndex()}  {self.modelsCombo.currentText()}")


    def read_data(self):
        print("load data")
        files = os.listdir(f"{os.getcwd()}//dane")
        time, value_temp, value_hum, value_acid, value_PV =[],[],[],[],[]
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



    def plot_data(self):
        if self.oneFileDict is not None:
            t = np.arange(0, len(self.oneFileDict["time"]))
            plt.plot(t, self.oneFileDict["value_temp"], c='b')
            plt.plot(t, self.oneFileDict["value_hum"], c='g')
            plt.plot(t, self.oneFileDict["value_acid"], c='y')
            plt.plot(t, self.oneFileDict["value_PV"], c='r')
            plt.grid(True)
            plt.show()


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




