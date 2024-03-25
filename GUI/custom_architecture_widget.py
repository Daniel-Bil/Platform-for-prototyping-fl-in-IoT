from functools import partial

import keras
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QLineEdit
from GUI.custom_layer_widget import LayerWidget
from GUI.utils import clearsubLayout

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

class ArchitectureWidget(QWidget):
    def __init__(self):
        super(ArchitectureWidget, self).__init__()
        self.setMinimumHeight(400)
        self.setMinimumWidth(400)
        self.setStyleSheet("background-color: #F67280;")
        self.mainLayout = QVBoxLayout()
        self.mainLayout.setAlignment(Qt.AlignTop)
        self.setLayout(self.mainLayout)
        self.test1()



    def test1(self):
        new_layer = LayerWidget()
        self.mainLayout.addWidget(new_layer)
        horizontal = QHBoxLayout()
        newbutton1 = QPushButton("add new layer")
        newbutton2 = QPushButton("save model")
        horizontal.addWidget(newbutton1)
        horizontal.addWidget(newbutton2)
        newbutton1.clicked.connect(self.del_button)
        newbutton1.clicked.connect(self.test1)
        self.mainLayout.addLayout(horizontal)

    def del_button(self):
        idx = self.mainLayout.count() - 1
        item = self.mainLayout.itemAt(idx)

        item2 = item.itemAt(1)
        item2.widget().deleteLater()
        item2 = item.itemAt(0)
        item2.widget().deleteLater()
        clearmainLayout(item)


    def create_emptyLayer(self):
        l = QHBoxLayout()
        label1 = QLabel("Add Layer")
        button1 = QPushButton("add Layer hehe")
        button1.clicked.connect(partial(clearmainLayout, self.mainLayout))
        button1.clicked.connect(self.create_filledLayer)
        button1.clicked.connect(self.create_emptyLayer)
        l.addWidget(label1)
        l.addWidget(button1)
        self.mainLayout.addLayout(l)
        print(f"num = {self.mainLayout.count()} create_emptyLayer")


    def create_filledLayer(self):
        num_elements = self.mainLayout.count()
        print(f"num = {num_elements}")
        if num_elements == 0:
            l = QHBoxLayout()
            combo1 = QComboBox()
            combo1.addItems(activation_functions)

            qline1 = QLineEdit("input size")


            l.addWidget(combo1)
            l.addWidget(qline1)
            self.mainLayout.addLayout(l)
            print(f" num = {self.mainLayout.count()} create_filledLayer")
        else:
            l = QHBoxLayout()
            combo1 = QComboBox()
            combo1.addItems(activation_functions)

            qline1 = QLineEdit("value")

            l.addWidget(combo1)
            l.addWidget(qline1)
            self.mainLayout.addLayout(l)
            print(f" num = {self.mainLayout.count()} create_filledLayer")


    def finish_layer(self):
        pass


def clearmainLayout(layout):

    if layout is not None:
        # Loop backwards (using negative indexing) to remove items from the end
        idx = layout.count()-1
        print(f"idx = {idx} before del {idx+1}")
        item = layout.itemAt(idx)
        if item.widget():
            item.widget().deleteLater()
        elif item.layout():
            clearsubLayout(item)
            layout.removeItem(item)
        else:
            raise Exception("something wrong in deleting layouts")
        print(f"afeter delidx = {layout.count()}")





