from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QComboBox, QVBoxLayout, QLabel, QLineEdit

from GUI.utils import clearsubLayout

common_layers = [
    "None",
    "Dense",
    "Conv2D",
    "MaxPooling2D",
    "Flatten",
    "Dropout"
]

activation_functions = [
    "linear",
    "relu",
    "sigmoid",
    "tanh",
    "softmax",
    "softplus",
    "softsign",
    "selu",
    "elu",
    "exponential"
]
class LayerWidget(QWidget):
    def __init__(self):
        super(LayerWidget, self).__init__()
        self.mainLayout = QHBoxLayout()
        self.mainLayout.setAlignment(Qt.AlignTop)
        self.setLayout(self.mainLayout)
        self.setStyleSheet("background-color: #F67280;")
        self.setMinimumHeight(50)
        self.setMinimumWidth(100)
        self.firstl()

    def firstl(self):
        print("firstl")
        self.combo1 = QComboBox()
        self.combo1.addItems(common_layers)
        self.mainLayout.addWidget(self.combo1)
        self.combo1.currentTextChanged.connect(self.clear_layout)
        self.combo1.currentTextChanged.connect(self.choose_parameters)


    def choose_parameters(self):
        print("chose parameters")
        chosen_layer = self.combo1.currentText()
        print(chosen_layer)
        if chosen_layer == "Dense":
            self.DenseUI()
        elif chosen_layer == "Conv2D":
            self.Conv2DUi()
        elif chosen_layer == "MaxPooling2D":
            pass
        elif chosen_layer == "Flatten":
            pass
        elif chosen_layer == "Dropout":
            pass
        else:
            pass


    def DenseUI(self):
        ll = ["units", "activ"]
        for l in ll:
            vertical = QVBoxLayout()
            vertical.setAlignment(Qt.AlignTop)
            label = QLabel(l)
            if l =="units":
                secondwidget = QLineEdit("0")
            elif l =="activ":
                secondwidget = QComboBox()
                secondwidget.addItems(activation_functions)
            else:
                raise Exception("There should be case for that")
            vertical.addWidget(label)
            vertical.addWidget(secondwidget)
            self.mainLayout.addLayout(vertical)
        button2 = QPushButton("specific")
        self.mainLayout.addWidget(button2)


    def Conv2DUi(self):
        ll = ["filters", "activ", "kernel_size"]
        for l in ll:
            print(l)
            vertical = QVBoxLayout()
            vertical.setAlignment(Qt.AlignTop)
            label = QLabel(l)
            print("1")
            if l == "filters":
                secondwidget = QLineEdit("0")
            elif l == "activ":
                secondwidget = QComboBox()
                secondwidget.addItems(activation_functions)
            elif l == "kernel_size":
                secondwidget = QLineEdit("0")
            else:
                raise Exception("There should be case for that")
            print("2")
            try:
                vertical.addWidget(label)
                vertical.addWidget(secondwidget)
            except Exception as e:
                print(e)
            print("3")
            self.mainLayout.addLayout(vertical)
        button2 = QPushButton("specific")
        print("4")
        self.mainLayout.addWidget(button2)


    def clear_layout(self):
        if self.mainLayout is not None:
            for i in range(self.mainLayout.count() - 1, -1, -1):
                if not i == 0:
                    item = self.mainLayout.itemAt(i)
                    if item.widget():
                        item.widget().deleteLater()
                    elif item.layout():
                        clearsubLayout(item)
                        self.mainLayout.removeItem(item)
                    else:
                        raise Exception("something wrong in deleting layouts")

