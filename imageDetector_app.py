import sys

from PyQt5.QtGui import QPixmap, QIcon

from src import MainWindow, RES_DIR
from PyQt5.QtWidgets import QApplication


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()

    pixmap = QPixmap(f'{RES_DIR}/app_icon.ico')
    icon = QIcon(pixmap)
    app.setWindowIcon(icon)

    # Set maximized screen mode
    main_window.showMaximized()
    sys.exit(app.exec_())
