from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from PySide6.QtCore import Qt, QEvent
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QComboBox, QLabel
from GUI.image_widget import ImageWidget
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


class SemiSupervisedWidget(QWidget):
    def __init__(self, better_data):
        super(SemiSupervisedWidget, self).__init__()
        self.window = 40
        self.big_window = 500

        self.better_data = better_data

        self.setMinimumHeight(800)
        self.setMinimumWidth(2000)
        self.setMaximumWidth(2000)
        self.setStyleSheet("background-color: #F67280;")
        self.mainLayout = QVBoxLayout()
        self.mainLayout.setAlignment(Qt.AlignTop)
        self.setLayout(self.mainLayout)
        self.current_data = self.better_data[0]

        self.indexes = []

        self.layout2 = QVBoxLayout()
        self.mainLayout.addLayout(self.layout2)

        self.figures1 = []
        self.canvases1 = []

        self.colors = {"time": (240/255,240/255,240/255),
                       "value_temp": (248/255,196/255,113/255),
                       "value_hum": (84/255,153/255,199/255),
                       "value_acid": (34/255,153/255,84/255),
                       "value_PV": (125/255,102/255,8/255)}

        for i in range(4):
            fig = Figure(figsize=(2000/100, 100/100), dpi=100)
            canv = FigureCanvas(fig)
            self.layout2.addWidget(canv)
            self.figures1.append(fig)
            self.canvases1.append(canv)



        self.layout3 = QHBoxLayout()
        self.mainLayout.addLayout(self.layout3)

        self.figures2 = []
        self.canvases2 = []

        for i in range(4):
            fig = Figure(figsize=(500 / 100, 200 / 100), dpi=100)
            canv = FigureCanvas(fig)
            self.layout3.addWidget(canv)
            self.figures2.append(fig)
            self.canvases2.append(canv)





        self.horizontalLayout1 = QHBoxLayout()
        self.horizontalLayout2 = QHBoxLayout()

        self.mainLayout.addLayout(self.horizontalLayout1)
        self.mainLayout.addLayout(self.horizontalLayout2)

        self.qline = QLineEdit("idx")
        self.qline2 = QLineEdit("0")
        self.horizontalLayout2.addWidget(self.qline)

        self.label = QLabel("label")
        self.label2 = QLabel("label2")

        self.jumpButton = QPushButton("jump")
        self.labelButton = QPushButton("addlabel")
        self.sampleButton = QPushButton("create\nsamples")


        self.jumpButton.clicked.connect(self.jump)
        self.labelButton.clicked.connect(self.add_label)
        self.sampleButton.clicked.connect(self.csamples)
        self.horizontalLayout2.addWidget(self.jumpButton)
        self.horizontalLayout2.addWidget(self.labelButton)
        self.horizontalLayout2.addWidget(self.sampleButton)

        self.combobox = QComboBox()
        self.combobox2 = QComboBox()
        for i in range(len(self.better_data)):
            self.combobox.addItem(f"{i}")

        self.combobox2.addItems(["value_temp", "value_hum", "value_acid", "value_PV"])

        self.horizontalLayout2.addWidget(self.combobox)
        self.horizontalLayout2.addWidget(self.combobox2)

        self.combobox.currentTextChanged.connect(self.change_current_data)
        self.combobox2.currentTextChanged.connect(self.update_images)

        self.buttons = []
        for i in range(6):
            if i == 0:
                button = QPushButton("1🢂")
            if i == 1:
                button = QPushButton("10🢂")
            if i == 2:
                button = QPushButton("100🢂")
            if i == 3:
                button = QPushButton("🢀1")
            if i == 4:
                button = QPushButton("🢀10")
            if i == 5:
                button = QPushButton("🢀100")
            button.clicked.connect(partial(self.jump_button, i))
            self.buttons.append(button)
            self.horizontalLayout2.addWidget(button)

        self.button_start = QPushButton("start")
        self.button_end = QPushButton("end")
        self.horizontalLayout2.addWidget(self.button_start)
        self.horizontalLayout2.addWidget(self.button_end)

        self.mainLayout.addWidget(self.label)
        self.mainLayout.addWidget(self.label2)
        self.mainLayout.addWidget(self.qline2)

        self.start_idx = 0
        self.end_idx = self.window

        self.start_idx2 = 0
        self.end_idx2 = self.big_window
        self.setMouseTracking(True)
        self.installEventFilter(self)
        self.update_images()





    def change_current_data(self):
        idx = self.combobox.currentIndex()
        self.current_data = self.better_data[idx]
        self.update_images()


    def update_images(self):
        print(self.start_idx, self.end_idx)
        self.label.setText(f"start: {self.start_idx} end: {self.end_idx} middle: {self.end_idx-(self.window//2)}")

        for fig in self.figures1:
            fig.clear()

        for fig in self.figures2:
            fig.clear()

        self.axes = []

        for i in range(8):
            if i < 4:
                ax = self.figures1[i].add_subplot(111)
            else:
                ax = self.figures2[i-4].add_subplot(111)
            ax.grid(True)
            ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

            self.axes.append(ax)
        keys = ["value_temp", "value_hum", "value_acid", "value_PV"]










        for ax, key in zip(self.axes, keys):
            ax.plot(self.current_data.iot_dict[key][0:self.start_idx], color='blue' if not self.combobox2.currentText()==key else "yellow", rasterized=True)
            ax.plot(range(self.start_idx, self.end_idx), self.current_data.iot_dict[key][self.start_idx:self.end_idx], color='green', rasterized=True)
            ax.plot(range(self.end_idx, len(self.current_data.iot_dict[key])), self.current_data.iot_dict[key][self.end_idx:], color='blue' if not self.combobox2.currentText()==key else "yellow", rasterized=True)

        #plot 4 clipped segments
        for ax, key in zip(self.axes[4:], keys):
            ax.plot(self.current_data.iot_dict[key][self.start_idx:self.end_idx], color='green' if not self.combobox2.currentText()==key else "yellow", rasterized=True)

        # plot error for keys
        for ax, key in zip(self.axes, keys):
            for index in self.current_data.errors[key]:
                ax.scatter(index, self.current_data.iot_dict[key][index], c=self.colors[key])

        else:
            for ax, key in zip(self.axes, keys):
                for index in self.current_data.errors["time"]:
                    ax.scatter(index, self.current_data.iot_dict[key][index], c=self.colors["time"])


        #polot errors for clipped plots
        for ax, key in zip(self.axes[4:], keys):
            for index in self.current_data.errors[key]:
                if self.start_idx < index < self.end_idx:
                    print("iundex", index)
                    ax.scatter(index-self.start_idx, self.current_data.iot_dict[key][index], c=self.colors[key])

        else:
            for ax, key in zip(self.axes[4:], keys):
                for index in self.current_data.errors["time"]:
                    if self.start_idx < index < self.end_idx:
                        print("iundex", index)
                        ax.scatter(index-self.start_idx, self.current_data.iot_dict[key][index], c=self.colors["time"])







        for canvas in self.canvases1:
            canvas.draw()
        for canvas in self.canvases2:
            canvas.draw()

    def eventFilter(self, source, event):
        # Check if the event is a mouse move event
        if event.type() == QEvent.MouseMove:
            # Update the label with the new cursor position
            pos = event.pos()
            self.label2.setText(f"Cursor Position: {pos.x()}, {pos.y()}")
        return super().eventFilter(source, event)

    def jump(self):
        text = self.qline.text()
        text = int(text)
        self.start_idx = text
        self.end_idx = self.start_idx+self.window

        self.update_images()

    def jump_button(self, id):
        if id == 0:
            jump = 1
        elif id == 1:
            jump = 10
        elif id == 2:
            jump = 100
        elif id == 3:
            jump = -1
        elif id == 4:
            jump = -10
        elif id == 5:
            jump = -100
        else:
            raise Exception("there shouldn't be this id")

        self.start_idx += jump
        self.end_idx = self.start_idx + self.window
        self.update_images()


    def add_label(self):
        print(f"add label on index = {self.start_idx+int(self.window//2)}")

        text = self.combobox2.currentText()

        self.current_data.errors[text].append(self.start_idx+int(self.window//2))
        self.update_images()

    def csamples2(self):
        for data in self.better_data:
            data.create_samples()

    def csamples(self):
        minmaxvalues = []
        for data in self.better_data:
            min_max = data.return_min_max()
            minmaxvalues.append(min_max)

        mins = {"value_temp": np.min([m["min"]["value_temp"] for m in minmaxvalues]),
                "value_hum": np.min([m["min"]["value_hum"] for m in minmaxvalues]),
                "value_acid": np.min([m["min"]["value_acid"] for m in minmaxvalues])}

        maxs = {"value_temp": np.max([m["max"]["value_temp"] for m in minmaxvalues]),
                "value_hum": np.max([m["max"]["value_hum"] for m in minmaxvalues]),
                "value_acid": np.max([m["max"]["value_acid"] for m in minmaxvalues])}




        for data in self.better_data:
            data.create_samples_normalised(mins=mins,maxs=maxs)

        print("brek")

