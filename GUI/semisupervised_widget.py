import matplotlib.pyplot as plt
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton
from GUI.image_widget import ImageWidget
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


class SemiSupervisedWidget(QWidget):
    def __init__(self, data):
        super(SemiSupervisedWidget, self).__init__()
        self.window = 40
        self.data = data["value_temp"]
        self.setMinimumHeight(800)
        self.setMinimumWidth(1200)
        self.setMaximumWidth(1200)
        self.setStyleSheet("background-color: #F67280;")
        self.mainLayout = QVBoxLayout()
        self.mainLayout.setAlignment(Qt.AlignTop)
        self.setLayout(self.mainLayout)


        self.indexes = []

        self.figure1 = Figure(figsize=(1000/100, 300/100), dpi=100)
        self.canvas1 = FigureCanvas(self.figure1)
        self.layout2 = QVBoxLayout()
        self.layout2.addWidget(self.canvas1)
        self.mainLayout.addLayout(self.layout2)
        self.ax1 = self.figure1.add_subplot(111)

        self.figure2 = Figure(figsize=(500 / 100, 300 / 100), dpi=100)
        self.canvas2 = FigureCanvas(self.figure2)
        self.layout3 = QVBoxLayout()
        self.layout3.addWidget(self.canvas2)
        self.mainLayout.addLayout(self.layout3)
        self.ax2 = self.figure2.add_subplot(111)

        self.horizontalLayout1 = QHBoxLayout()
        self.horizontalLayout2 = QHBoxLayout()

        self.mainLayout.addLayout(self.horizontalLayout1)
        self.mainLayout.addLayout(self.horizontalLayout2)

        self.qline = QLineEdit("idx")
        self.horizontalLayout2.addWidget(self.qline)

        self.jumpButton = QPushButton("jump")
        self.labelButton = QPushButton("addlabel")
        self.jumpButton.clicked.connect(self.jump)
        self.labelButton.clicked.connect(self.add_label)
        self.horizontalLayout2.addWidget(self.jumpButton)
        self.horizontalLayout2.addWidget(self.labelButton)

        self.jumpButton1 = QPushButton("10🢂")
        self.jumpButton2 = QPushButton("100🢂")
        self.jumpButton3 = QPushButton("🢀100")
        self.jumpButton4 = QPushButton("🢀10")

        self.jumpButton1.clicked.connect(self.jump1)
        self.jumpButton2.clicked.connect(self.jump2)
        self.jumpButton3.clicked.connect(self.jump3)
        self.jumpButton4.clicked.connect(self.jump4)



        self.horizontalLayout2.addWidget(self.jumpButton3)
        self.horizontalLayout2.addWidget(self.jumpButton4)
        self.horizontalLayout2.addWidget(self.jumpButton1)
        self.horizontalLayout2.addWidget(self.jumpButton2)
        self.start_idx = 0
        self.end_idx = self.window

        self.update_images()


    def update_images(self):
        print(self.start_idx, self.end_idx)

        self.figure1.clear()
        self.figure2.clear()

        self.ax1 = self.figure1.add_subplot(111)  # Recreate ax1
        self.ax2 = self.figure2.add_subplot(111)  # Recreate ax2

        self.ax1.plot(self.data[0:self.start_idx], color='blue')
        self.ax1.plot(range(self.start_idx,self.end_idx),self.data[self.start_idx:self.end_idx], color='green')
        self.ax1.plot(range(self.end_idx,len(self.data)),self.data[self.end_idx:], color='blue')

        for index in self.indexes:
            self.ax1.scatter(index,self.data[index],c="red")

        self.ax2.plot(self.data[self.start_idx:self.end_idx], color='green')
        for index in self.indexes:
            if self.start_idx < index < self.end_idx:
                self.ax2.scatter(self.end_idx-index,self.data[index],c="red")
        self.canvas1.draw()
        self.canvas2.draw()


    def jump(self):
        text = self.qline.text()
        text = int(text)
        self.start_idx = text
        self.end_idx = self.start_idx+self.window

        self.update_images()

    def jump1(self):
        self.start_idx += 10
        self.end_idx = self.start_idx+self.window
        self.update_images()

    def jump2(self):
        self.start_idx += 100
        self.end_idx = self.start_idx+self.window
        self.update_images()

    def jump3(self):
        self.start_idx -= 100
        self.end_idx = self.start_idx+self.window
        self.update_images()

    def jump4(self):
        self.start_idx -= 10
        self.end_idx = self.start_idx+self.window
        self.update_images()

    def add_label(self):
        self.indexes.append(self.start_idx+20)
        self.update_images()


