import numpy as np
import tensorflow as tf

from PySide6.QtWidgets import QMainWindow


class PlatformWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Platform for prototyping federated learning in IoT")
        self.setGeometry(50, 50, 1600, 800)


    def set_layout(self):
        pass

