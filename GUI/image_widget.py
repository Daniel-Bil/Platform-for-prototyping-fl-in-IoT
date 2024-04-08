import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QImage, QColor
from PySide6.QtWidgets import QWidget
import matplotlib.pyplot as plt


def qImageToNumpyArray(qimage):
    """Convert QImage to a numpy array."""

    # Get QImage dimensions
    width, height = qimage.width(), qimage.height()

    # QImage.Format.Format_RGB32 means each pixel is 32 bits
    if qimage.format() in [QImage.Format.Format_ARGB32, QImage.Format.Format_RGB32]:
        # Convert to an array
        ptr = qimage.bits()
        ptr.setsize(qimage.byteCount())
        arr = np.array(ptr).reshape(height, width, 4)  # For ARGB32, 4 channels
    elif qimage.format() == QImage.Format.Format_RGB888:
        # Convert to an array
        ptr = qimage.bits()
        ptr.setsize(qimage.byteCount())
        arr = np.array(ptr).reshape(height, width, 3)  # For RGB888, 3 channels
    else:
        # Implement conversion for other formats as needed
        raise ValueError("Unsupported QImage format")

    return arr

class ImageWidget(QWidget):
    def __init__(self,parent=None, size = (400, 400)):
        super(ImageWidget, self).__init__(parent)
        self.setFixedSize(size[0],size[1])
        # self.setMinimumHeight(size[0])
        # self.setMinimumWidth(size[1])
        # self.setMaximumHeight(size[0])
        # self.setMaximumWidth(size[1])
        self.image = QImage(size[1], size[0], QImage.Format_RGB32)
        gray_color = QColor(128, 128, 128)  # RGB color for gray
        self.image.fill(gray_color)


    def update_image(self, image):
        self.image = image
        self.update()

    def paintEvent(self, event):

        painter = QPainter(self)
        # Draw the image at the top-left corner
        painter.drawImage(0, 0, self.image.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

