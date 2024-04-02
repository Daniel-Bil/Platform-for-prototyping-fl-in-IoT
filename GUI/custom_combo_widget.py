from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QComboBox

class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        self.mainComboBox = QComboBox()
        self.mainComboBox.addItems(['Option 1', 'Option 2', 'Option 3'])
        self.mainComboBox.currentIndexChanged.connect(self.onMainComboChanged)
        layout.addWidget(self.mainComboBox)

        self.secondaryComboBox = QComboBox()
        layout.addWidget(self.secondaryComboBox)

        self.onMainComboChanged(self.mainComboBox.currentIndex())

    def onMainComboChanged(self, index):
        # Clear current items in secondary combo box
        self.secondaryComboBox.clear()

        # Based on the selection in the main combo box, add specific items to the secondary combo box
        if index == 0:  # Option 1
            self.secondaryComboBox.addItems(['1.1', '1.2', '1.3'])
        elif index == 1:  # Option 2
            self.secondaryComboBox.addItems(['2.1', '2.2', '2.3'])
        elif index == 2:  # Option 3
            self.secondaryComboBox.addItems(['3.1', '3.2', '3.3'])