"""

Qline edits and labels informing how many decvices takees part in learning -> custom label + edit widgets


"""



from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGridLayout

from GUI.custom_qlineedit import CustomLabelEdit

"""

Grid layout in widget with 4 x 4 grid of Edits

"""

class ParametersHandler(QWidget):
    def __init__(self):
        super(ParametersHandler, self).__init__()
        self.setMinimumHeight(450)
        self.setMinimumWidth(450)
        self.mainGridLayout = QGridLayout()
        self.setLayout(self.mainGridLayout)
        self.customEdit1 = CustomLabelEdit("Number of sensors")
        self.customEdit2 = CustomLabelEdit("parameter2")
        self.customEdit3 = CustomLabelEdit("parameter3")

        self.customEdit4 = CustomLabelEdit("Number of epochs")
        self.customEdit5 = CustomLabelEdit("parameter5")
        self.customEdit6 = CustomLabelEdit("parameter6")

        self.customEdit7 = CustomLabelEdit("parameter7")
        self.customEdit8 = CustomLabelEdit("parameter8")
        self.customEdit9 = CustomLabelEdit("parameter9")

        self.mainGridLayout.addWidget(self.customEdit1, 0, 0)
        self.mainGridLayout.addWidget(self.customEdit2, 0, 1)
        self.mainGridLayout.addWidget(self.customEdit3, 0, 2)

        self.mainGridLayout.addWidget(self.customEdit4, 1, 0)
        self.mainGridLayout.addWidget(self.customEdit5, 1, 1)
        self.mainGridLayout.addWidget(self.customEdit6, 1, 2)

        self.mainGridLayout.addWidget(self.customEdit7, 2, 0)
        self.mainGridLayout.addWidget(self.customEdit8, 2, 1)
        self.mainGridLayout.addWidget(self.customEdit9, 2, 2)



