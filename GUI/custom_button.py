
from PySide6.QtWidgets import QPushButton
import os


class CustomQPushButton(QPushButton):
    def __init__(self, text=""):
        super(CustomQPushButton, self).__init__(text)
        with open(f"{os.getcwd()}//GUI//custombutton.stylesheet") as file:
            self.setStyleSheet(file.read())

        self.setMinimumWidth(150)
        self.setMinimumHeight(50)

        self.setMaximumWidth(150)
        self.setMaximumHeight(50)
