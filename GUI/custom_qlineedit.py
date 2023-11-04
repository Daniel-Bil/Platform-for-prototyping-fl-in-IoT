

from PySide6.QtWidgets import QLabel, QLineEdit, QWidget, QHBoxLayout


class CustomLabelEdit(QWidget):
    def __init__(self, text="label"):
        super(CustomLabelEdit,self).__init__()
        self.mainLayout = QHBoxLayout()
        self.setLayout(self.mainLayout)

        self.label1 = QLabel(text)
        self.label1.setMaximumHeight(50)
        self.label1.setStyleSheet("background-color: #00ccff;  color:#ffffff; font-size: 20px; border-radius: 5px;")
        self.qedit1 = QLineEdit("0")
        self.qedit1.setStyleSheet("background-color: #ffffff;")

        self.mainLayout.addWidget(self.label1)
        self.mainLayout.addWidget(self.qedit1)



