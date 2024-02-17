
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGridLayout

from GUI.custom_button import CustomQPushButton

"""

Grid layout in widget with 4 x 4 grid of buttons

"""

class ButtonMenuHandler(QWidget):
    def __init__(self):
        super(ButtonMenuHandler, self).__init__()
        self.setMinimumHeight(450)
        self.setMinimumWidth(450)
        self.mainGridLayout = QGridLayout()
        self.setLayout(self.mainGridLayout)
        self.customButton1 = CustomQPushButton("Load model")
        self.customButton2 = CustomQPushButton("Load data")
        self.customButton3 = CustomQPushButton("Plot data")

        self.customButton4 = CustomQPushButton("find good")
        self.customButton5 = CustomQPushButton("plot good")
        self.customButton6 = CustomQPushButton("find empty")

        self.customButton7 = CustomQPushButton("connect timeseries")
        self.customButton8 = CustomQPushButton("8")
        self.customButton9 = CustomQPushButton("show results")

        self.mainGridLayout.addWidget(self.customButton1, 0, 0)
        self.mainGridLayout.addWidget(self.customButton2, 0, 1)
        self.mainGridLayout.addWidget(self.customButton3, 0, 2)

        self.mainGridLayout.addWidget(self.customButton4, 1, 0)
        self.mainGridLayout.addWidget(self.customButton5, 1, 1)
        self.mainGridLayout.addWidget(self.customButton6, 1, 2)

        self.mainGridLayout.addWidget(self.customButton7, 2, 0)
        self.mainGridLayout.addWidget(self.customButton8, 2, 1)
        self.mainGridLayout.addWidget(self.customButton9, 2, 2)



