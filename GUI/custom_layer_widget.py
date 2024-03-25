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
            self.MaxPooling2DUI()
        elif chosen_layer == "Flatten":
            self.FlattenUI()
        elif chosen_layer == "Dropout":
            self.DropoutUI()
        else:
            pass
        if not chosen_layer=="None":
            self.add_end(self.mainLayout)


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


    def Conv2DUi(self):
        ll = ["filters", "activ", "kernel_size"]
        for l in ll:
            vertical = QVBoxLayout()
            vertical.setAlignment(Qt.AlignTop)
            label = QLabel(l)
            if l == "filters":
                secondwidget = QLineEdit("0")
            elif l == "activ":
                secondwidget = QComboBox()
                secondwidget.addItems(activation_functions)
            elif l == "kernel_size":
                secondwidget = QLineEdit("0")
            else:
                raise Exception("There should be case for that")
            vertical.addWidget(label)
            vertical.addWidget(secondwidget)

            self.mainLayout.addLayout(vertical)


    def add_end(self, layout):
        button2 = QPushButton("specific")
        button2.setMaximumWidth(50)
        layout.addWidget(button2)
        button3 = QPushButton("delete")
        button3.setMaximumWidth(50)
        layout.addWidget(button3)
        button4 = QPushButton(u"insert \u2193")
        button4.setMaximumWidth(50)
        layout.addWidget(button4)

        button3.clicked.connect(self.handle_deletion)
        button4.clicked.connect(self.handle_insert)

    def handle_deletion(self):
        self.setParent(None)
        self.deleteLater()

    def handle_insert(self):
        parentLayout = self.parent().layout()  # Assuming the parent widget is set and has a layout
        index = parentLayout.indexOf(self)
        newWidget = LayerWidget()
        parentLayout.insertWidget(index + 1, newWidget)

    def FlattenUI(self):
        pass

    def DropoutUI(self):
        vertical = QVBoxLayout()
        vertical.setAlignment(Qt.AlignTop)
        label = QLabel("rate")
        secondwidget = QLineEdit("0")
        vertical.addWidget(label)
        vertical.addWidget(secondwidget)

        self.mainLayout.addLayout(vertical)


    def MaxPooling2DUI(self):
        ll = ["pool_size", "strides", "padding"]
        for l in ll:
            vertical = QVBoxLayout()
            vertical.setAlignment(Qt.AlignTop)
            label = QLabel(l)
            if l == "pool_size":
                secondwidget = QLineEdit("0")
            elif l == "strides":
                secondwidget = QLineEdit("0")
            elif l == "padding":
                secondwidget = QComboBox()
                secondwidget.addItems(["valid", "same"])
            else:
                raise Exception("There should be case for that")
            vertical.addWidget(label)
            vertical.addWidget(secondwidget)

            self.mainLayout.addLayout(vertical)

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

