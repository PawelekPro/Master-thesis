import sys

from src import MainWindow
from PyQt5.QtWidgets import QApplication


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()

    # Set maximized screen mode
    main_window.showMaximized()
    sys.exit(app.exec_())