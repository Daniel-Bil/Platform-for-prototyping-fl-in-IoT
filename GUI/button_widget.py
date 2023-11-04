
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
        self.customButton1 = CustomQPushButton("1")
        self.customButton2 = CustomQPushButton("2")
        self.customButton3 = CustomQPushButton("3")

        self.customButton4 = CustomQPushButton("4")
        self.customButton5 = CustomQPushButton("5")
        self.customButton6 = CustomQPushButton("6")

        self.customButton7 = CustomQPushButton("7")
        self.customButton8 = CustomQPushButton("8")
        self.customButton9 = CustomQPushButton("9")

        self.mainGridLayout.addWidget(self.customButton1, 0, 0)
        self.mainGridLayout.addWidget(self.customButton2, 0, 1)
        self.mainGridLayout.addWidget(self.customButton3, 0, 2)

        self.mainGridLayout.addWidget(self.customButton4, 1, 0)
        self.mainGridLayout.addWidget(self.customButton5, 1, 1)
        self.mainGridLayout.addWidget(self.customButton6, 1, 2)

        self.mainGridLayout.addWidget(self.customButton7, 2, 0)
        self.mainGridLayout.addWidget(self.customButton8, 2, 1)
        self.mainGridLayout.addWidget(self.customButton9, 2, 2)



