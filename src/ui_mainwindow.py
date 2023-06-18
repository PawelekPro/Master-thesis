# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'forms/mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1074, 1025)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_9 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_9.setContentsMargins(0, 0, 5, 0)
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.image_view_tab = QtWidgets.QWidget()
        font = QtGui.QFont()
        font.setFamily("Open Sans")
        self.image_view_tab.setFont(font)
        self.image_view_tab.setObjectName("image_view_tab")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.image_view_tab)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.img_view_layout = QtWidgets.QGridLayout()
        self.img_view_layout.setObjectName("img_view_layout")
        self.ren_win_1 = QtWidgets.QLabel(self.image_view_tab)
        self.ren_win_1.setText("")
        self.ren_win_1.setScaledContents(True)
        self.ren_win_1.setObjectName("ren_win_1")
        self.img_view_layout.addWidget(self.ren_win_1, 0, 0, 1, 1)
        self.ren_win_2 = QtWidgets.QLabel(self.image_view_tab)
        self.ren_win_2.setText("")
        self.ren_win_2.setScaledContents(True)
        self.ren_win_2.setObjectName("ren_win_2")
        self.img_view_layout.addWidget(self.ren_win_2, 0, 1, 1, 1)
        self.ren_win_4 = QtWidgets.QLabel(self.image_view_tab)
        self.ren_win_4.setText("")
        self.ren_win_4.setScaledContents(True)
        self.ren_win_4.setObjectName("ren_win_4")
        self.img_view_layout.addWidget(self.ren_win_4, 1, 1, 1, 1)
        self.ren_win_3 = QtWidgets.QLabel(self.image_view_tab)
        self.ren_win_3.setText("")
        self.ren_win_3.setScaledContents(True)
        self.ren_win_3.setObjectName("ren_win_3")
        self.img_view_layout.addWidget(self.ren_win_3, 1, 0, 1, 1)
        self.gridLayout_2.addLayout(self.img_view_layout, 0, 0, 1, 1)
        self.tabWidget.addTab(self.image_view_tab, "")
        self.postprocessing = QtWidgets.QWidget()
        self.postprocessing.setObjectName("postprocessing")
        self.gridLayout_10 = QtWidgets.QGridLayout(self.postprocessing)
        self.gridLayout_10.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_10.setObjectName("gridLayout_10")
        self.pltWidget = matplotlibWidget(self.postprocessing)
        self.pltWidget.setObjectName("pltWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.pltWidget)
        self.gridLayout.setObjectName("gridLayout")
        self.gridLayout_10.addWidget(self.pltWidget, 0, 0, 1, 1)
        self.tabWidget.addTab(self.postprocessing, "")
        self.gridLayout_9.addWidget(self.tabWidget, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1074, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.option_control = QtWidgets.QDockWidget(MainWindow)
        self.option_control.setMinimumSize(QtCore.QSize(440, 836))
        font = QtGui.QFont()
        font.setFamily("Open Sans")
        self.option_control.setFont(font)
        self.option_control.setFloating(False)
        self.option_control.setFeatures(QtWidgets.QDockWidget.AllDockWidgetFeatures)
        self.option_control.setObjectName("option_control")
        self.dockWidgetContents = QtWidgets.QWidget()
        font = QtGui.QFont()
        font.setFamily("Open Sans")
        self.dockWidgetContents.setFont(font)
        self.dockWidgetContents.setObjectName("dockWidgetContents")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.dockWidgetContents)
        self.gridLayout_4.setContentsMargins(2, 2, 2, 2)
        self.gridLayout_4.setHorizontalSpacing(6)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.advanced_options_gbox = QtWidgets.QGroupBox(self.dockWidgetContents)
        self.advanced_options_gbox.setMinimumSize(QtCore.QSize(0, 0))
        self.advanced_options_gbox.setMaximumSize(QtCore.QSize(16777215, 180))
        self.advanced_options_gbox.setCheckable(True)
        self.advanced_options_gbox.setObjectName("advanced_options_gbox")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.advanced_options_gbox)
        self.gridLayout_5.setContentsMargins(5, 5, 5, 5)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.on_fly_preview = QtWidgets.QCheckBox(self.advanced_options_gbox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.on_fly_preview.sizePolicy().hasHeightForWidth())
        self.on_fly_preview.setSizePolicy(sizePolicy)
        self.on_fly_preview.setMinimumSize(QtCore.QSize(0, 20))
        self.on_fly_preview.setMaximumSize(QtCore.QSize(16777215, 20))
        self.on_fly_preview.setText("")
        self.on_fly_preview.setChecked(True)
        self.on_fly_preview.setObjectName("on_fly_preview")
        self.verticalLayout_5.addWidget(self.on_fly_preview)
        self.make_gif = QtWidgets.QCheckBox(self.advanced_options_gbox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.make_gif.sizePolicy().hasHeightForWidth())
        self.make_gif.setSizePolicy(sizePolicy)
        self.make_gif.setMinimumSize(QtCore.QSize(0, 20))
        self.make_gif.setMaximumSize(QtCore.QSize(16777215, 20))
        self.make_gif.setText("")
        self.make_gif.setChecked(False)
        self.make_gif.setObjectName("make_gif")
        self.verticalLayout_5.addWidget(self.make_gif)
        self.fps = QtWidgets.QSpinBox(self.advanced_options_gbox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fps.sizePolicy().hasHeightForWidth())
        self.fps.setSizePolicy(sizePolicy)
        self.fps.setMaximumSize(QtCore.QSize(16777215, 20))
        self.fps.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.fps.setFrame(True)
        self.fps.setAlignment(QtCore.Qt.AlignCenter)
        self.fps.setMinimum(50)
        self.fps.setMaximum(50000)
        self.fps.setProperty("value", 2000)
        self.fps.setObjectName("fps")
        self.verticalLayout_5.addWidget(self.fps)
        self.gridLayout_5.addLayout(self.verticalLayout_5, 0, 1, 1, 1)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_9 = QtWidgets.QLabel(self.advanced_options_gbox)
        self.label_9.setMinimumSize(QtCore.QSize(335, 0))
        self.label_9.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label_9.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_9.setMidLineWidth(0)
        self.label_9.setIndent(5)
        self.label_9.setObjectName("label_9")
        self.verticalLayout_4.addWidget(self.label_9)
        self.label_12 = QtWidgets.QLabel(self.advanced_options_gbox)
        self.label_12.setMinimumSize(QtCore.QSize(250, 0))
        self.label_12.setMaximumSize(QtCore.QSize(16666666, 20))
        self.label_12.setIndent(5)
        self.label_12.setObjectName("label_12")
        self.verticalLayout_4.addWidget(self.label_12)
        self.label_11 = QtWidgets.QLabel(self.advanced_options_gbox)
        self.label_11.setMinimumSize(QtCore.QSize(250, 0))
        self.label_11.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label_11.setIndent(5)
        self.label_11.setObjectName("label_11")
        self.verticalLayout_4.addWidget(self.label_11)
        self.gridLayout_5.addLayout(self.verticalLayout_4, 0, 0, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.path_to_save = QtWidgets.QPushButton(self.advanced_options_gbox)
        self.path_to_save.setObjectName("path_to_save")
        self.horizontalLayout_2.addWidget(self.path_to_save)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.stop_button = QtWidgets.QPushButton(self.advanced_options_gbox)
        self.stop_button.setMaximumSize(QtCore.QSize(50, 16777215))
        self.stop_button.setObjectName("stop_button")
        self.horizontalLayout_2.addWidget(self.stop_button)
        self.start_conv = QtWidgets.QPushButton(self.advanced_options_gbox)
        self.start_conv.setMaximumSize(QtCore.QSize(50, 16777215))
        self.start_conv.setObjectName("start_conv")
        self.horizontalLayout_2.addWidget(self.start_conv)
        self.gridLayout_5.addLayout(self.horizontalLayout_2, 3, 0, 1, 2)
        self.label_10 = QtWidgets.QLabel(self.advanced_options_gbox)
        self.label_10.setMinimumSize(QtCore.QSize(208, 15))
        self.label_10.setMaximumSize(QtCore.QSize(0, 10))
        font = QtGui.QFont()
        font.setItalic(False)
        self.label_10.setFont(font)
        self.label_10.setIndent(5)
        self.label_10.setObjectName("label_10")
        self.gridLayout_5.addWidget(self.label_10, 1, 0, 1, 1)
        self.path_line_edit = QtWidgets.QLineEdit(self.advanced_options_gbox)
        self.path_line_edit.setFrame(True)
        self.path_line_edit.setReadOnly(True)
        self.path_line_edit.setObjectName("path_line_edit")
        self.gridLayout_5.addWidget(self.path_line_edit, 2, 0, 1, 2)
        self.gridLayout_3.addWidget(self.advanced_options_gbox, 2, 0, 1, 1)
        self.source_gbox = QtWidgets.QGroupBox(self.dockWidgetContents)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.source_gbox.sizePolicy().hasHeightForWidth())
        self.source_gbox.setSizePolicy(sizePolicy)
        self.source_gbox.setMinimumSize(QtCore.QSize(0, 100))
        self.source_gbox.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.source_gbox.setFlat(False)
        self.source_gbox.setCheckable(False)
        self.source_gbox.setObjectName("source_gbox")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.source_gbox)
        self.gridLayout_6.setContentsMargins(5, 5, 5, 5)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.browse_file_button = QtWidgets.QPushButton(self.source_gbox)
        self.browse_file_button.setObjectName("browse_file_button")
        self.gridLayout_6.addWidget(self.browse_file_button, 1, 1, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(210, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_6.addItem(spacerItem1, 1, 0, 1, 1)
        self.file_list = QtWidgets.QTreeView(self.source_gbox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.file_list.sizePolicy().hasHeightForWidth())
        self.file_list.setSizePolicy(sizePolicy)
        self.file_list.setMinimumSize(QtCore.QSize(0, 100))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.file_list.setFont(font)
        self.file_list.setFrameShape(QtWidgets.QFrame.Box)
        self.file_list.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.file_list.setLineWidth(1)
        self.file_list.setObjectName("file_list")
        self.gridLayout_6.addWidget(self.file_list, 0, 0, 1, 2)
        self.gridLayout_3.addWidget(self.source_gbox, 0, 0, 1, 1)
        self.options_gbox = QtWidgets.QGroupBox(self.dockWidgetContents)
        self.options_gbox.setMinimumSize(QtCore.QSize(0, 280))
        self.options_gbox.setMaximumSize(QtCore.QSize(16777215, 280))
        self.options_gbox.setCheckable(True)
        self.options_gbox.setObjectName("options_gbox")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.options_gbox)
        self.gridLayout_7.setContentsMargins(5, 5, 5, 5)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label = QtWidgets.QLabel(self.options_gbox)
        self.label.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label.setIndent(5)
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.label_2 = QtWidgets.QLabel(self.options_gbox)
        self.label_2.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label_2.setIndent(6)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_2.addWidget(self.label_2)
        self.label_3 = QtWidgets.QLabel(self.options_gbox)
        self.label_3.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label_3.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_3.setMidLineWidth(0)
        self.label_3.setIndent(5)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_2.addWidget(self.label_3)
        self.label_7 = QtWidgets.QLabel(self.options_gbox)
        self.label_7.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label_7.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_7.setMidLineWidth(0)
        self.label_7.setIndent(5)
        self.label_7.setObjectName("label_7")
        self.verticalLayout_2.addWidget(self.label_7)
        self.label_4 = QtWidgets.QLabel(self.options_gbox)
        self.label_4.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label_4.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_4.setMidLineWidth(0)
        self.label_4.setIndent(5)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_2.addWidget(self.label_4)
        self.label_5 = QtWidgets.QLabel(self.options_gbox)
        self.label_5.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label_5.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_5.setMidLineWidth(0)
        self.label_5.setIndent(5)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_2.addWidget(self.label_5)
        self.label_6 = QtWidgets.QLabel(self.options_gbox)
        self.label_6.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label_6.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_6.setMidLineWidth(0)
        self.label_6.setIndent(5)
        self.label_6.setObjectName("label_6")
        self.verticalLayout_2.addWidget(self.label_6)
        self.label_8 = QtWidgets.QLabel(self.options_gbox)
        self.label_8.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label_8.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_8.setMidLineWidth(0)
        self.label_8.setIndent(5)
        self.label_8.setObjectName("label_8")
        self.verticalLayout_2.addWidget(self.label_8)
        self.gridLayout_7.addLayout(self.verticalLayout_2, 0, 0, 1, 2)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.threshold = QtWidgets.QSpinBox(self.options_gbox)
        self.threshold.setMaximumSize(QtCore.QSize(16777215, 20))
        self.threshold.setFrame(True)
        self.threshold.setMaximum(255)
        self.threshold.setProperty("value", 25)
        self.threshold.setObjectName("threshold")
        self.verticalLayout_3.addWidget(self.threshold)
        self.ground_height = QtWidgets.QSpinBox(self.options_gbox)
        self.ground_height.setMaximumSize(QtCore.QSize(16777215, 20))
        self.ground_height.setFrame(True)
        self.ground_height.setMaximum(1024)
        self.ground_height.setProperty("value", 475)
        self.ground_height.setObjectName("ground_height")
        self.verticalLayout_3.addWidget(self.ground_height)
        self.interpolation_range = QtWidgets.QSpinBox(self.options_gbox)
        self.interpolation_range.setMaximumSize(QtCore.QSize(16777215, 20))
        self.interpolation_range.setFrame(True)
        self.interpolation_range.setMaximum(120)
        self.interpolation_range.setProperty("value", 40)
        self.interpolation_range.setObjectName("interpolation_range")
        self.verticalLayout_3.addWidget(self.interpolation_range)
        self.interpolation_deg = QtWidgets.QSpinBox(self.options_gbox)
        self.interpolation_deg.setMaximumSize(QtCore.QSize(16777215, 20))
        self.interpolation_deg.setFrame(True)
        self.interpolation_deg.setMinimum(2)
        self.interpolation_deg.setMaximum(7)
        self.interpolation_deg.setProperty("value", 2)
        self.interpolation_deg.setObjectName("interpolation_deg")
        self.verticalLayout_3.addWidget(self.interpolation_deg)
        self.raw_image = QtWidgets.QCheckBox(self.options_gbox)
        self.raw_image.setMinimumSize(QtCore.QSize(0, 20))
        self.raw_image.setMaximumSize(QtCore.QSize(16777215, 20))
        self.raw_image.setText("")
        self.raw_image.setChecked(True)
        self.raw_image.setObjectName("raw_image")
        self.verticalLayout_3.addWidget(self.raw_image)
        self.raw_contour = QtWidgets.QCheckBox(self.options_gbox)
        self.raw_contour.setMinimumSize(QtCore.QSize(0, 20))
        self.raw_contour.setMaximumSize(QtCore.QSize(16777215, 20))
        self.raw_contour.setText("")
        self.raw_contour.setChecked(True)
        self.raw_contour.setObjectName("raw_contour")
        self.verticalLayout_3.addWidget(self.raw_contour)
        self.result_data = QtWidgets.QCheckBox(self.options_gbox)
        self.result_data.setMinimumSize(QtCore.QSize(0, 20))
        self.result_data.setMaximumSize(QtCore.QSize(16777215, 20))
        self.result_data.setText("")
        self.result_data.setChecked(True)
        self.result_data.setObjectName("result_data")
        self.verticalLayout_3.addWidget(self.result_data)
        self.invert = QtWidgets.QComboBox(self.options_gbox)
        self.invert.setObjectName("invert")
        self.invert.addItem("")
        self.invert.addItem("")
        self.verticalLayout_3.addWidget(self.invert)
        self.gridLayout_7.addLayout(self.verticalLayout_3, 0, 2, 1, 1)
        self.preview_button = QtWidgets.QPushButton(self.options_gbox)
        self.preview_button.setObjectName("preview_button")
        self.gridLayout_7.addWidget(self.preview_button, 1, 2, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(210, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_7.addItem(spacerItem2, 1, 0, 1, 1)
        self.gridLayout_3.addWidget(self.options_gbox, 1, 0, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout_3, 0, 0, 1, 1)
        self.option_control.setWidget(self.dockWidgetContents)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.option_control)
        self.outputTerminal = QtWidgets.QDockWidget(MainWindow)
        self.outputTerminal.setMinimumSize(QtCore.QSize(440, 144))
        font = QtGui.QFont()
        font.setFamily("Open Sans")
        self.outputTerminal.setFont(font)
        self.outputTerminal.setObjectName("outputTerminal")
        self.terminal_widget = QtWidgets.QWidget()
        self.terminal_widget.setStyleSheet("")
        self.terminal_widget.setObjectName("terminal_widget")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.terminal_widget)
        self.gridLayout_8.setContentsMargins(5, 0, 0, 0)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.output_panel = QtWidgets.QTextEdit(self.terminal_widget)
        self.output_panel.setMinimumSize(QtCore.QSize(435, 0))
        self.output_panel.setMaximumSize(QtCore.QSize(16777215, 200))
        self.output_panel.setStyleSheet("")
        self.output_panel.setFrameShape(QtWidgets.QFrame.Box)
        self.output_panel.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.output_panel.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContentsOnFirstShow)
        self.output_panel.setTabChangesFocus(False)
        self.output_panel.setReadOnly(True)
        self.output_panel.setObjectName("output_panel")
        self.gridLayout_8.addWidget(self.output_panel, 0, 0, 1, 1)
        self.outputTerminal.setWidget(self.terminal_widget)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.outputTerminal)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.image_view_tab), _translate("MainWindow", "Image view"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.postprocessing), _translate("MainWindow", "Postprocessing"))
        self.option_control.setWindowTitle(_translate("MainWindow", "Option control"))
        self.advanced_options_gbox.setTitle(_translate("MainWindow", "Advanced Options"))
        self.label_9.setText(_translate("MainWindow", "Preview during conversion"))
        self.label_12.setText(_translate("MainWindow", "Make GIF"))
        self.label_11.setText(_translate("MainWindow", "FPS "))
        self.path_to_save.setText(_translate("MainWindow", "Browse"))
        self.stop_button.setText(_translate("MainWindow", "Stop"))
        self.start_conv.setText(_translate("MainWindow", "Start"))
        self.label_10.setText(_translate("MainWindow", "Path to save data:"))
        self.source_gbox.setTitle(_translate("MainWindow", "Source control"))
        self.browse_file_button.setText(_translate("MainWindow", "Browse"))
        self.options_gbox.setTitle(_translate("MainWindow", "Graphical Options"))
        self.label.setText(_translate("MainWindow", "Threshold (binary)"))
        self.label_2.setText(_translate("MainWindow", "Ground height [px]"))
        self.label_3.setText(_translate("MainWindow", "Interpolation range [px]"))
        self.label_7.setText(_translate("MainWindow", "Interp. polynomials degree"))
        self.label_4.setText(_translate("MainWindow", "Display raw image"))
        self.label_5.setText(_translate("MainWindow", "Display raw contour"))
        self.label_6.setText(_translate("MainWindow", "Display result image"))
        self.label_8.setText(_translate("MainWindow", "Invert result image"))
        self.invert.setItemText(0, _translate("MainWindow", "No"))
        self.invert.setItemText(1, _translate("MainWindow", "Yes"))
        self.preview_button.setText(_translate("MainWindow", "Preview"))
        self.preview_button.setShortcut(_translate("MainWindow", "Return"))
        self.outputTerminal.setWindowTitle(_translate("MainWindow", "Output terminal"))
from src.matplotlibWidget import matplotlibWidget