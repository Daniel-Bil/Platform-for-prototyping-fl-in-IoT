import numpy as np
import tensorflow as tf
# from tensorflow.keras.applications import ResNet50 <- sprawic by bylo zainstalowane tam sa modele
from PySide6.QtGui import Qt

from PySide6.QtWidgets import QMainWindow, QHBoxLayout, QVBoxLayout, QWidget, QLabel, QComboBox
import keras

from GUI.button_widget import ButtonMenuHandler
from GUI.custom_button import CustomQPushButton
from GUI.parameters_widget import ParametersHandler


class PlatformWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Platform for prototyping federated learning in IoT")
        self.setGeometry(50, 50, 1600, 800)
        self.set_layout()
        self.setStyleSheet("background-color: #669999;")


    def load_model(self):
        print(f"{self.modelsCombo.currentIndex()}  {self.modelsCombo.currentText()}")



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

