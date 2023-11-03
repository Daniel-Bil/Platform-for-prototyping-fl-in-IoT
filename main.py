import sys

from PySide6.QtWidgets import QApplication
from main_platform import PlatformWindow



if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = PlatformWindow()
    window.show()

    app.exec()
