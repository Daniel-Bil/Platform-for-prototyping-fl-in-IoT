import numpy as np
import tensorflow as tf
# from tensorflow.keras.applications import ResNet50 <- sprawic by bylo zainstalowane tam sa modele
from PySide6.QtGui import Qt

from PySide6.QtWidgets import QMainWindow, QHBoxLayout, QVBoxLayout, QPushButton, QWidget, QLabel, QComboBox
import keras

class PlatformWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Platform for prototyping federated learning in IoT")
        self.setGeometry(50, 50, 1600, 800)
        self.set_layout()



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
        self.horizontalLayout3 = QHBoxLayout()

        self.mainLayout.addLayout(self.horizontalLayout1)
        self.mainLayout.addLayout(self.horizontalLayout2)
        self.mainLayout.addLayout(self.horizontalLayout3)

        self.verticalLayout1 = QVBoxLayout()
        self.verticalLayout2 = QVBoxLayout()
        self.verticalLayout1.setAlignment(Qt.AlignLeft)
        self.verticalLayout2.setAlignment(Qt.AlignRight)
        self.horizontalLayout2.addLayout(self.verticalLayout1)
        self.horizontalLayout2.addLayout(self.verticalLayout2)

        self.label1 = QLabel("models: ")
        self.modelsCombo = QComboBox()
        self.modelsCombo.addItem("Model Test")
        self.modelsCombo.addItem("Model Test2")
        self.horizontalLayout1.addWidget(self.label1)
        self.horizontalLayout1.addWidget(self.modelsCombo)

        self.b1 = QPushButton("Load model form combo")
        self.b2 = QPushButton("2")
        self.verticalLayout2.addWidget(self.b1)
        self.verticalLayout2.addWidget(self.b2)


        self.b1.clicked.connect(self.load_model)
