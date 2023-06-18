import os

from PyQt5.QtWidgets import QMessageBox
from src.mainwindow import MainWindow
import src.utils

ACTION_CANCEL = QMessageBox.StandardButton.Cancel
ACTION_SAVE = QMessageBox.StandardButton.Save
ACTION_NO = QMessageBox.StandardButton.No
ACTION_OK = QMessageBox.StandardButton.Ok

IMG_FILTER = 'PNG files (*.png, *.PNG)'
CSV_FILTER = 'CSV files (*.csv, *.CSV)'

PROJ_NAME = 'ImageDetector'
__src_path__ = os.path.dirname(__file__)
__proj_path__ = os.path.dirname(__src_path__)
RES_DIR = os.path.join(__proj_path__, 'resources')
ICONS_DIR = os.path.join(RES_DIR, 'icons')
TEMP_DIR = os.path.join(__proj_path__, 'temporary')

QSS_Q_TAB_WIDGET = \
            """
            QTabWidget::pane {
                border-radius: 25px;
                border: 1px solid lightgray;
                margin-top: 2px;
            }
            QTabBar::tab {
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                border-bottom-left-radius: 5px;
                background-color: lightgray;
                height: 25px;
                width: 150px;
            }
            QTabBar::tab:selected {
                background-color: white;
            }
            """

QSS_Q_CHILD_TAB_WIDGET = \
            """
            QTabWidget::pane {
                border-radius: 5px;
                border: 1px solid lightgray;
                margin-top: 2px;
            }
            QTabBar::tab {
                border-top-left-radius: 3px;
                border-top-right-radius: 3px;
                border-bottom-left-radius: 3px;
                background-color: lightgray;
                height: 15px;
                width: 100px;
            }
            QTabBar::tab:selected {
                background-color: white;
            }
            """

QSS_Q_LABEL = """
             QLabel {
             border-radius: 25px;
             background-color: lightgray;
                    }      
              """

QSS_Q_GROUP = """
              QGroupBox { 
              border-radius: 10px; 
              border: 1px solid lightgray;
              padding-top: 10px;
              }
              """

QSS_PROGRESS = """QProgressBar {
    border: 1px solid grey;
    border-radius: 5px;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #05B8CC;
    width: 20px;
}"""
