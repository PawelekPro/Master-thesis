# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'forms/plot_control.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_plotControl(object):
    def setupUi(self, plotControl):
        plotControl.setObjectName("plotControl")
        plotControl.resize(400, 200)
        plotControl.setMaximumSize(QtCore.QSize(16777214, 16777215))
        font = QtGui.QFont()
        font.setFamily("Open Sans")
        plotControl.setFont(font)
        self.gridLayout_2 = QtWidgets.QGridLayout(plotControl)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.gridLayout_3.setContentsMargins(10, 5, 10, 5)
        self.gridLayout_3.setVerticalSpacing(10)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label = QtWidgets.QLabel(plotControl)
        self.label.setMaximumSize(QtCore.QSize(16777215, 10))
        self.label.setIndent(0)
        self.label.setObjectName("label")
        self.gridLayout_3.addWidget(self.label, 5, 0, 1, 1)
        self.line_width = QtWidgets.QDoubleSpinBox(plotControl)
        self.line_width.setDecimals(2)
        self.line_width.setMinimum(0.25)
        self.line_width.setMaximum(5.0)
        self.line_width.setSingleStep(0.25)
        self.line_width.setProperty("value", 0.5)
        self.line_width.setObjectName("line_width")
        self.gridLayout_3.addWidget(self.line_width, 1, 1, 1, 1)
        self.plot_button = QtWidgets.QPushButton(plotControl)
        self.plot_button.setObjectName("plot_button")
        self.gridLayout_3.addWidget(self.plot_button, 6, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(plotControl)
        self.label_2.setObjectName("label_2")
        self.gridLayout_3.addWidget(self.label_2, 3, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(plotControl)
        self.label_3.setObjectName("label_3")
        self.gridLayout_3.addWidget(self.label_3, 1, 0, 1, 1)
        self.title_line = QtWidgets.QLineEdit(plotControl)
        self.title_line.setMinimumSize(QtCore.QSize(0, 20))
        self.title_line.setMaximumSize(QtCore.QSize(16777215, 20))
        self.title_line.setObjectName("title_line")
        self.gridLayout_3.addWidget(self.title_line, 6, 0, 1, 1)
        self.avg_scope = QtWidgets.QSpinBox(plotControl)
        self.avg_scope.setMinimum(1)
        self.avg_scope.setMaximum(100)
        self.avg_scope.setProperty("value", 10)
        self.avg_scope.setObjectName("avg_scope")
        self.gridLayout_3.addWidget(self.avg_scope, 3, 1, 1, 1)
        self.groupBox = QtWidgets.QGroupBox(plotControl)
        self.groupBox.setCheckable(False)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout.setObjectName("gridLayout")
        self.label_4 = QtWidgets.QLabel(self.groupBox)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 0, 0, 1, 1)
        self.dl_min = QtWidgets.QSpinBox(self.groupBox)
        self.dl_min.setMaximumSize(QtCore.QSize(65, 16777215))
        self.dl_min.setMinimum(-100)
        self.dl_min.setMaximum(100)
        self.dl_min.setProperty("value", -2)
        self.dl_min.setObjectName("dl_min")
        self.gridLayout.addWidget(self.dl_min, 0, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.groupBox)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 1, 0, 1, 1)
        self.dl_max = QtWidgets.QSpinBox(self.groupBox)
        self.dl_max.setMinimum(-100)
        self.dl_max.setMaximum(100)
        self.dl_max.setProperty("value", 30)
        self.dl_max.setObjectName("dl_max")
        self.gridLayout.addWidget(self.dl_max, 1, 1, 1, 1)
        self.gridLayout_3.addWidget(self.groupBox, 4, 0, 1, 2)
        self.gridLayout_2.addLayout(self.gridLayout_3, 1, 0, 1, 1)

        self.retranslateUi(plotControl)
        QtCore.QMetaObject.connectSlotsByName(plotControl)

    def retranslateUi(self, plotControl):
        _translate = QtCore.QCoreApplication.translate
        plotControl.setWindowTitle(_translate("plotControl", "GroupBox"))
        self.label.setText(_translate("plotControl", "Plot title:"))
        self.plot_button.setText(_translate("plotControl", "Plot"))
        self.plot_button.setShortcut(_translate("plotControl", "P"))
        self.label_2.setText(_translate("plotControl", "Moving average scope [frames]"))
        self.label_3.setText(_translate("plotControl", "Line width [px]"))
        self.groupBox.setTitle(_translate("plotControl", "Contact length growth (dL) range"))
        self.label_4.setText(_translate("plotControl", "Min [px]"))
        self.label_5.setText(_translate("plotControl", "Max [px]"))
