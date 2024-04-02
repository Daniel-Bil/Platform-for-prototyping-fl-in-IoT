from PySide6 import QtCore
from PySide6.QtWidgets import QLabel, QLineEdit, QWidget, QHBoxLayout


class CustomLabelEdit(QWidget):
    def __init__(self, text="label"):
        super(CustomLabelEdit,self).__init__()
        self.mainLayout = QHBoxLayout()
        self.setLayout(self.mainLayout)

        self.label1 = QLabel(text)
        self.label1.setMaximumHeight(50)
        self.label1.setStyleSheet("background-color: #518dbf;  color:#ffffff; font-size: 20px; font: bold; border-radius: 5px;")
        self.label1.setAlignment(QtCore.Qt.AlignCenter)
        self.qedit1 = QLineEdit("0")
        self.qedit1.setMaximumWidth(100)
        self.qedit1.setMinimumHeight(70)
        self.qedit1.setAlignment(QtCore.Qt.AlignCenter)
        self.qedit1.setStyleSheet("background-color: #ffffff; font-size: 20px; border-radius: 5px;")

        self.mainLayout.addWidget(self.label1)
        self.mainLayout.addWidget(self.qedit1)


class CustomLabelEdit2(QWidget):
    def __init__(self, text="label", parent=None):
        super(CustomLabelEdit2, self).__init__(parent)
        self.mainLayout = QHBoxLayout()
        self.setLayout(self.mainLayout)

        self.label1 = QLabel(text)
        self.label1.setMaximumHeight(30)
        self.label1.setStyleSheet("background-color: #C06C84;  color:#ffffff; font-size: 20px; font: bold; border-radius: 5px;")
        self.label1.setAlignment(QtCore.Qt.AlignCenter)
        self.qedit1 = QLineEdit("0")
        self.qedit1.setMaximumWidth(50)
        self.qedit1.setMinimumHeight(30)
        self.qedit1.setMaximumHeight(30)
        self.qedit1.setAlignment(QtCore.Qt.AlignCenter)
        self.qedit1.setStyleSheet("background-color: #ffffff; font-size: 20px; border-radius: 5px;")

        self.mainLayout.addWidget(self.label1)
        self.mainLayout.addWidget(self.qedit1)



