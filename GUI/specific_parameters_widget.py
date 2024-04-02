import math

from PySide6.QtCore import Qt
from PySide6.QtGui import QRegion, QPainter, QMouseEvent, QColor, QPen, QBrush, QRadialGradient
from PySide6.QtWidgets import QWidget
from PySide6.QtWidgets import QLabel, QLineEdit

from GUI.custom_qlineedit import CustomLabelEdit2


class FloatingWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(0, 0, 400, 400)  # Position and size
        self.setMask(QRegion(self.rect(), QRegion.Ellipse))  # Circular shape
        self.w1 = CustomLabelEdit2("par1", self)
        self.w2 = CustomLabelEdit2("par2", self)
        self.w3 = CustomLabelEdit2("par3", self)
        self.w4 = QLineEdit("2", self)
        self.w5 = QLineEdit("2", self)
        self.w6 = QLineEdit("2", self)
        self.w7 = QLineEdit("2", self)
        self.w8 = QLineEdit("2", self)
        self.w9 = QLineEdit("2", self)




        self.layoutWidgetsInCircle([self.w1,self.w2,self.w3,self.w4,self.w5,self.w6,self.w7,self.w8,self.w9],100)

    def layoutWidgetsInCircle(self, widgets, radius):
        center_x = self.width() / 2
        center_y = self.height() / 2
        angle_increment = 360 / len(widgets)

        for i, widget in enumerate(widgets):
            angle = math.radians(angle_increment * i)
            widget_x = center_x + radius * math.cos(angle) - (widget.width() / 2)
            widget_y = center_y + radius * math.sin(angle) - (widget.height() / 2)
            widget.move(int(widget_x), int(widget_y))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Create a radial gradient centered in the widget
        gradient = QRadialGradient(self.rect().center(), self.rect().width() / 2)
        gradient.setColorAt(0, QColor("#6C5B7B"))  # Inner color
        gradient.setColorAt(1, QColor("#F8B195"))  # Outer color

        # Set the gradient as the brush
        painter.setBrush(QBrush(gradient))
        painter.setPen(QPen(Qt.NoPen))  # No border
        painter.drawEllipse(self.rect())  # Draw the ellipse with the gradient brush

    def mousePressEvent(self, event: QMouseEvent):
        self.offset = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() == Qt.LeftButton:
            # Calculate the new position of the window.
            new_position = event.globalPosition().toPoint() - self.offset
            self.move(new_position)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        self.isPressed = False
        super().mouseReleaseEvent(event)