import matplotlib.pyplot as plt
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton
from GUI.image_widget import ImageWidget

class SemiSupervisedWidget(QWidget):
    def __init__(self, data):
        super(SemiSupervisedWidget, self).__init__()
        self.data = data["value_temp"]
        self.setMinimumHeight(800)
        self.setMinimumWidth(1000)
        self.setStyleSheet("background-color: #F67280;")
        self.mainLayout = QVBoxLayout()
        self.mainLayout.setAlignment(Qt.AlignTop)
        self.setLayout(self.mainLayout)


        self.imgw1 = ImageWidget(size=(1000,200))
        self.mainLayout.addWidget(self.imgw1)


        self.horizontalLayout1 = QHBoxLayout()
        self.horizontalLayout2 = QHBoxLayout()

        self.mainLayout.addLayout(self.horizontalLayout1)
        self.mainLayout.addLayout(self.horizontalLayout2)

        self.imgw2 = ImageWidget(size=(400,200))
        self.horizontalLayout1.addWidget(self.imgw2)

        self.qline = QLineEdit("idx")
        self.horizontalLayout2.addWidget(self.qline)

        self.jumpButton = QPushButton("jump")
        self.jumpButton.clicked.connect(self.jump)
        self.horizontalLayout2.addWidget(self.jumpButton)

        self.start_idx = 0
        self.end_idx = 30

        self.update_images()


    def update_images(self):
        print(self.start_idx, self.end_idx)
        plt.figure()
        plt.plot(self.data[0:self.start_idx], color='blue')
        plt.plot(self.data[self.start_idx:self.end_idx], color='green')
        plt.plot(self.data[self.end_idx:], color='blue')
        plt.draw()
        canvas = plt.gcf().canvas
        canvas.draw()
        width, height = canvas.get_width_height()

        # Step 3: Convert the canvas to a string (buffer), then to a NumPy array
        buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        buf.shape = (height, width, 3)

        # Step 4: Convert the NumPy array to QImage
        qimage = QImage(buf, width, height, QImage.Format_RGB888)
        new_width = 1000
        new_height = 200

        # Resize the image
        qimage = qimage.scaled(new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.imgw1.update_image(qimage)
        # Use qimage as needed in your PyQt application...

        # Don't forget to close the plot if you're not showing it
        plt.close()

        plt.figure()
        plt.plot(self.data[self.start_idx:self.end_idx], color='blue')

        plt.draw()
        canvas = plt.gcf().canvas
        canvas.draw()
        width, height = canvas.get_width_height()

        # Step 3: Convert the canvas to a string (buffer), then to a NumPy array
        buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        buf.shape = (height, width, 3)

        # Step 4: Convert the NumPy array to QImage
        qimage = QImage(buf, width, height, QImage.Format_RGB888)
        new_width = 400
        new_height = 200

        # Resize the image
        qimage = qimage.scaled(new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.imgw2.update_image(qimage)
        # Use qimage as needed in your PyQt application...

        # Don't forget to close the plot if you're not showing it
        plt.close()

    def jump(self):
        text = self.qline.text()
        text = int(text)
        self.start_idx = text
        self.end_idx = self.start_idx+30

        self.update_images()


    def add_label(self):
        pass



