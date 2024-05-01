import sys

from PySide6.QtWidgets import QApplication
from main_platform import PlatformWindow
from logic.dataProcesing import hello_world
from logic.dataProcesing import preprocess
if __name__ == "__main__":
    #print(hello_world())
    #meine()
    app = QApplication(sys.argv)

    window = PlatformWindow()
    window.show()

    app.exec()

