from PyQt5.QtWidgets import QGroupBox

from src.ui_fftcontrol import Ui_fftControl
from src.ui_plotcontrol import Ui_plotControl


class plotControl(QGroupBox, Ui_plotControl):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)


class fftControl(QGroupBox, Ui_fftControl):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)