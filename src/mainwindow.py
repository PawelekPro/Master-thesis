import csv
import glob
import math
import os
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Callable, Tuple, Any, List

import cv2
import numpy as np
from PIL import Image
from PyQt5.QtCore import Qt, QCoreApplication, QSize
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox, QFileSystemModel, QWidget, QLabel, QVBoxLayout, \
    QProgressBar, QSizePolicy, QStatusBar, QDockWidget, QSplitter, QGroupBox, QGridLayout, QSpacerItem, \
    QPushButton, QTreeView, QFrame, QTabBar

import src
from src.interface_setup import plotControl, fftControl
from src.matplotlibWidget import fft_graphic_widget, matplotlibWidget
from src.postprocessing.csvReader import csvReader
from src.postprocessing.fft_analysis import FFTAnalysis
from src.ui_mainwindow import Ui_MainWindow
from src.utils import create_rounded_pixmap, signature, printText
from src.utils.utilities import draw_ruler


class MainWindow(QMainWindow, Ui_MainWindow):
    _project_directory: str = ''
    _data_path: str = ''
    _project_name: str = ''
    _csv_data_path: str = ''
    _running: bool = False
    progressBar: QProgressBar = None
    _label_left: list = []
    _label_right: list = []

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.add_progress_bar()

        self.file_model = QFileSystemModel()
        self.file_model.setRootPath("")
        self.file_list.setModel(self.file_model)
        self.file_list.setRootIndex(self.file_model.index(""))

        self.post_dock_widget = QDockWidget("Postprocessing control", self)
        self.set_post_widget()
        self.post_dock_widget.close()

        self.__set_connections()

    def __set_connections(self) -> None:
        self.tabWidget.setStyleSheet(src.QSS_Q_TAB_WIDGET)

        self.source_gbox.setStyleSheet(src.QSS_Q_GROUP)
        self.options_gbox.setStyleSheet(src.QSS_Q_GROUP)
        self.advanced_options_gbox.setStyleSheet(src.QSS_Q_GROUP)

        self.browse_file_button.clicked.connect(
            lambda: self.open_file_dialog(
                self.load_data, 'Load images', src.IMG_FILTER))

        self.preview_button.clicked.connect(self.preview)
        self.path_to_save.clicked.connect(lambda: self.open_file_dialog(
            self.directory_valid, 'Path to save', src.IMG_FILTER))

        self.start_conv.clicked.connect(self.run_conversion)
        self.stop_button.clicked.connect(self.stop_conversion)
        self.tabWidget.currentChanged.connect(self.state_changed)
        self.plot_options_gbox.plot_button.clicked.connect(self.on_plot_clicked)
        self.browse_csv.clicked.connect(lambda: self.open_file_dialog(
            self.load_csv, 'Load images', src.CSV_FILTER))

        self.fft_options_gbox.fft_plot_button.clicked.connect(
            self.plot_fft_clicked)
        self.fft_options_gbox.fft_clear_button.clicked.connect(
            self.on_clear_plot_clicked)

    @property
    def project_directory(self):
        """ Property for keeping the path of the output files. """
        return self._project_directory

    @property
    def data_path(self) -> str:
        """ Property for keeping the path of images directory. """
        return self._data_path

    @property
    def project_name(self) -> str:
        """ Property for setting the default name of output file. """
        return self._project_name

    @property
    def csv_data_path(self) -> str:
        """ Property for keeping the path of csv directory. """
        return self._csv_data_path

    @project_directory.setter
    def project_directory(self, path: str):
        self._project_directory = path

    @data_path.setter
    def data_path(self, path: str):
        self._data_path = path

    @project_name.setter
    def project_name(self, name: str):
        self._project_name = name

    @csv_data_path.setter
    def csv_data_path(self, name: str):
        self._csv_data_path = name

    def state_changed(self, index) -> None:
        current_tab = self.tabWidget.tabText(index)
        self.print_message(f'Current tab view changed to: {current_tab}')
        if current_tab == 'Postprocessing':
            self.removeDockWidget(self.outputTerminal)
            self.removeDockWidget(self.option_control)
            self.addDockWidget(Qt.LeftDockWidgetArea, self.post_dock_widget)
            self.addDockWidget(Qt.LeftDockWidgetArea, self.outputTerminal)
            self.post_dock_widget.show()
            self.outputTerminal.show()
        else:
            self.removeDockWidget(self.outputTerminal)
            self.removeDockWidget(self.post_dock_widget)
            self.addDockWidget(Qt.LeftDockWidgetArea, self.option_control)
            self.addDockWidget(Qt.LeftDockWidgetArea, self.outputTerminal)
            self.option_control.show()
            self.outputTerminal.show()

    def directory_valid(self, fname: str) -> None:
        if not os.path.exists(fname):
            os.makedirs(fname)

        # Set path to save output files
        self.project_directory = fname
        self.path_line_edit.setText(fname)

    def load_csv(self, fname: str) -> None:
        """Load directory with the given `fname`."""
        basename = os.path.basename(fname)

        if len(os.listdir(fname)) == 0:
            QMessageBox.warning(
                self, src.PROJ_NAME,
                f'The folder "{basename}" is empty.\n' +
                'Operation aborted.')
            return

        # Assign chosen path to data_path attribute and project_name
        self.csv_data_path = fname

        self.csv_file_model.setRootPath(fname)
        self.csv_file_model.setNameFilters(["*.csv"])
        self.csv_file_model.setNameFilterDisables(False)
        self.csv_list.setRootIndex(self.csv_file_model.index(fname))

    def load_data(self, fname: str) -> None:
        """Load directory with the given `fname`."""
        basename = os.path.basename(fname)

        if len(os.listdir(fname)) == 0:
            QMessageBox.warning(
                self, src.PROJ_NAME,
                f'The folder "{basename}" is empty.\n' +
                'Operation aborted.')
            return

        # Assign chosen path to data_path attribute and project_name
        self.data_path = fname
        self.project_name = basename

        self.file_model.setRootPath(fname)
        self.file_model.setNameFilters(["*.png"])
        self.file_model.setNameFilterDisables(False)
        self.file_list.setRootIndex(self.file_model.index(fname))

    def add_progress_bar(self):
        statusBar = QStatusBar()
        self.setStatusBar(statusBar)
        self.progressBar = QProgressBar()

        self.progressBar.setMaximum(100)
        self.progressBar.setMinimum(0)

        statusBar.addWidget(self.progressBar, 1)  # Add progress bar with stretch factor 1
        statusBar.setSizeGripEnabled(False)
        self.progressBar.setMaximumHeight(15)
        self.progressBar.setMaximumWidth(300)

        self.progressBar.setStyleSheet(src.QSS_PROGRESS)
        self.progressBar.hide()

    def stop_conversion(self):
        self._running = False

    def run_conversion(self) -> None:
        """ Runs the conversion process for all images available in chosen directory. """

        if os.path.exists(self.data_path) is False:
            QMessageBox.warning(
                self, src.PROJ_NAME,
                'First you have to select directory containing data.\n' +
                'Operation aborted.')
            return

        if len(os.listdir(self.data_path)) == 0:
            QMessageBox.warning(
                self, src.PROJ_NAME,
                f'The folder "{self.data_path}" is empty.\n' +
                'Operation aborted.')
            return

        if os.path.exists(self.project_directory) is False:
            QMessageBox.warning(
                self, src.PROJ_NAME,
                'You have to select directory to save the files.\n' +
                'Operation aborted.')
            return

        self._running = True
        self.output_panel.clear()
        self.print_message(f'Conversion process started: {datetime.now().time()}')
        self.progressBar.show()

        try:
            conv_data = self._run_conversion()
            if conv_data[0]:
                # csv writer
                self.export_to_csv(conv_data[0], conv_data[1], conv_data[2], conv_data[3], conv_data[4])

        except Exception as e:
            QMessageBox.warning(
                self, src.PROJ_NAME,
                f'Error while converting:\n' +
                f'{e}')
            self.print_message(f'\n\nERROR: Operation failed:\n {e}')

        self.progressBar.hide()

    def _run_conversion(self) -> Tuple[Any, List[Any], List[Any], List[Any]]:
        if os.path.exists(src.TEMP_DIR):
            shutil.rmtree(src.TEMP_DIR)
        os.mkdir(src.TEMP_DIR)

        frame = []
        paths = []
        for path in Path(self.data_path).glob("*.png"):
            frame.append(int(str(path)[-9] + str(path)[-8] + str(path)[-7] + str(path)[-6] + str(path)[-5]))
            paths.append(str(path))

        leftAngle = []
        rightAngle = []
        contactLength = []
        crossSectionArea = []
        for i in range(len(frame)):
            if not self._running:
                self.print_message(f'\n\nWARNING: Operation aborted by the user.')
                break
            self.print_message(f'{[i]} - Converting: {paths[i]}')
            self.progressBar.setValue(int(i / len(frame) * 100))

            ret_img = self.binary_threshold(paths[i])
            ret_img_copy = deepcopy(ret_img)
            cont_img, cont_obj = self.contour_detecting(ret_img_copy)
            result_img, ret_1, ret_2, ret_3, ret_4 = self.interpolate_polynomials(cont_img, cont_obj, True)
            leftAngle.append(ret_1)
            rightAngle.append(ret_2)
            contactLength.append(ret_3)
            contactLength.append(ret_3)
            crossSectionArea.append(ret_4)

            if self.on_fly_preview.isChecked():
                # Manage the existing widgets to avoid window size adjusting
                while self.img_view_layout.count():
                    item = self.img_view_layout.takeAt(0)
                    widget = item.layout()
                    if widget:
                        widget.setParent(None)
                        widget.deleteLater()
                self.set_photo(path=paths[i], qlabel=self.ren_win_1)
                self.set_photo(img=ret_img, qlabel=self.ren_win_2)
                self.set_photo(img=cont_img, qlabel=self.ren_win_3)
                self.set_photo(img=result_img, qlabel=self.ren_win_4)

            if self.make_gif.isChecked():
                path_to_save = f'{src.TEMP_DIR}/' + paths[i][-23:]
                cv2.imwrite(str(path_to_save), result_img)  # type: ignore

            # Process pending events to keep the GUI responsive
            QCoreApplication.processEvents()

        time = [frame_ / self.fps.value() for frame_ in frame]
        if self._running:
            self.progressBar.setValue(100)
            self.print_message(f'\n\nSuccessfully converted {len(paths)} files.')

            # Manage creating GIF - output
            if self.make_gif.isChecked():
                self.create_gif()
                self.print_message(f'\nGIF file has been created.')

            self.print_message(f'\nOutput folder: {self.project_directory}/{self.project_name}')

            return time, leftAngle, rightAngle, contactLength, crossSectionArea

        return [], [], [], []

    def export_to_csv(
            self, list_1: list, list_2: list, list_3: list, list_4: list, list_5: list) -> None:
        rows = zip(list_1, list_2, list_3, list_4, list_5)
        name = self.project_name
        with open(f'{self.project_directory}/{name}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Time', 'Left Angle', 'Right Angle', 'Contact Length', 'Cross Section Area'])
            writer.writerows(rows)

    def preview(self):
        first_index = self.file_model.index(0, 0, self.file_list.rootIndex())
        first_file_path = self.file_model.filePath(first_index)

        _, extension = os.path.splitext(first_file_path)
        if extension.lower() != '.png':
            QMessageBox.warning(
                self, src.PROJ_NAME,
                'First you have to select directory containing data.\n' +
                'Operation aborted.')
            return

        image_path = first_file_path
        selected_item = None
        if self.file_list.selectedIndexes():
            for index in self.file_list.selectedIndexes():
                selected_item = self.file_model.filePath(index)
            image_path = selected_item

        while self.img_view_layout.count():
            item = self.img_view_layout.takeAt(0)
            widget = item.layout()
            if widget:
                widget.setParent(None)
                widget.deleteLater()

        self.image_view_tab.setStyleSheet(src.QSS_Q_TAB_WIDGET)
        self.set_photo(path=image_path, qlabel=self.ren_win_1)

        ret_img = None
        if self.raw_image.isChecked():
            ret_img = self.binary_threshold(image_path)
            self.set_photo(qlabel=self.ren_win_2, img=ret_img)
        else:
            self.set_photo(f'{src.RES_DIR}/background.png', self.ren_win_2)

        cont_img = None
        if self.raw_contour.isChecked() and ret_img is not None:
            try:
                cont_img, contour = self.contour_detecting(ret_img)  # type: ignore
            except Exception as e:
                QMessageBox.warning(
                    self, src.PROJ_NAME,
                    f'Error while converting:\n' +
                    f'{e}')
                self.print_message(f'\n\nERROR: Operation failed:\n {e}')

            self.set_photo(qlabel=self.ren_win_3, img=cont_img)
        else:
            self.set_photo(f'{src.RES_DIR}/background.png', self.ren_win_3)

        result_img = None
        if self.result_data.isChecked() and cont_img is not None:
            try:
                result_img = self.interpolate_polynomials(cont_img, contour)  # noqa
            except Exception as e:
                QMessageBox.warning(
                    self, src.PROJ_NAME,
                    f'Error while converting:\n' +
                    f'{e}')
                self.print_message(f'\n\nERROR: Operation failed:\n {e}')

            if self.invert.currentText() == 'Yes':
                result_img = cv2.bitwise_not(result_img)  # type: ignore
            self.set_photo(qlabel=self.ren_win_4, img=result_img)
        else:
            self.set_photo(f'{src.RES_DIR}/background.png', self.ren_win_4)

    def open_file_dialog(
            self, action: Callable[[str], None],
            action_title: str, filter_: str) -> int:
        """Show file dialog to open the file and perform the *action*.

        :return: returns ACTION_OK if the action succeed or ACTION_NO
        otherwise.
        :rtype: int
        """
        fname = QFileDialog.getExistingDirectory(
            self, f'Select folder')

        if fname:
            action(fname)
            return src.ACTION_OK

        return src.ACTION_CANCEL

    @staticmethod
    def set_photo(
            path: str = None, qlabel: QLabel = None, img=None) -> None:
        if path is None:
            image = img
        else:
            image = cv2.imread(path)

        pixmap = create_rounded_pixmap(image, 15)
        qlabel.setPixmap(pixmap)

    def binary_threshold(self, path: str):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # type: ignore

        threshold = self.threshold.value()

        """ Binary thresholding """
        _, thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)  # type: ignore
        thresh = 255 - thresh

        return thresh

    def contour_detecting(self, img):
        # cutting out everything below the plate
        img[self.ground_height.value() + 1:len(img), :] = 0

        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # type: ignore
        filter_matrix = np.zeros_like(img)
        output_object = np.zeros_like(img)

        if len(contours) != 0:
            c_object = max(contours, key=cv2.contourArea)  # type: ignore
            cv2.drawContours(filter_matrix, c_object, -1, 255, 1)  # type: ignore
            cv2.fillPoly(filter_matrix, pts=[c_object], color=(255, 255, 255))  # type: ignore
            self.use_filter(filter_matrix, c_object, self.ground_height.value())  # type: ignore
            cont_filtered, _ = cv2.findContours(filter_matrix, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # type: ignore
            cv2.drawContours(output_object, cont_filtered, -1, 255, 1)  # type: ignore

        output_object[self.ground_height.value(), :] = 125
        # Draw ruler representing interpolation range
        draw_ruler(output_object, self.interpolation_range.value(), 5, (15, self.ground_height.value()))
        return output_object, cont_filtered

    @staticmethod
    def use_filter(img, contour, init_height):
        # Initialize min and max x-coordinates with the first point
        min_x = contour[0][0][0]
        max_x = contour[0][0][0]

        # Iterate over the remaining points and update min and max x-coordinates
        for point in contour:
            x = point[0][0]
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x

        height = init_height
        right_corn_coord = (max_x - 80, height - 50)

        # Todo: Think about how it should work using PointPolygonTest
        cv2.rectangle(img, (min_x + 80, init_height), right_corn_coord, 255, -1)  # type: ignore

    def interpolate_polynomials(self, img, contour_object, ret: bool = False):
        pixels = np.argwhere(img == 255)
        x = (pixels[:, 1])
        y = (pixels[:, 0])

        # Selecting set of points on which the spline will be spanned
        x, y = zip(*sorted(zip(x, y)))

        x_select = x[0:len(x):1]
        y_select = y[0:len(y):1]
        y_select = np.asarray(y_select)
        x_select = np.asarray(x_select)

        int_thresh = self.ground_height.value() - \
                     self.interpolation_range.value()

        y_cutoff_ids = np.where(y_select > int_thresh)
        x_split = x_select[np.argmin(y_select)]
        y_select = y_select[y_cutoff_ids]
        x_select = x_select[y_cutoff_ids]

        # split points to left and right groups in order to create separate polynomials
        idx_left = np.where(x_select < x_split)
        y_left = y_select[idx_left]
        x_left = x_select[idx_left]
        idx_right = np.where(x_select > x_split)
        y_right = y_select[idx_right]
        x_right = x_select[idx_right]

        z_left = np.polyfit(y_left, x_left, self.interpolation_deg.value())
        p_left = np.poly1d(z_left)

        z_right = np.polyfit(y_right, x_right, self.interpolation_deg.value())
        p_right = np.poly1d(z_right)

        # calculate derivatives and use them to get the angle
        dp_left = np.polyder(p_left, 1)
        angle_left = np.pi / 2 + np.arctan(dp_left(max(y_left) - 1))

        dp_right = np.polyder(p_right, 1)
        angle_right = np.pi - (np.pi / 2 + np.arctan(dp_right(max(y_right) - 1)))

        # draw tangent lines
        line_len = 150
        line_thickness = 1

        img_copy = np.zeros_like(img)
        p1 = np.array([x_left[np.argmax(y_left)], max(y_left)])
        p2 = p1 + np.array([line_len * np.cos(angle_left), -line_len * np.sin(angle_left)])
        cv2.line(img_copy, (p1[0], p1[1]), (round(p2[0]), round(p2[1])), 255, thickness=line_thickness)  # type: ignore

        p1 = np.array([x_right[np.argmax(y_right)], max(y_right)])
        p2 = p1 + np.array([-line_len * np.cos(angle_right), -line_len * np.sin(angle_right)])
        cv2.line(img_copy, (p1[0], p1[1]), (round(p2[0]), round(p2[1])), 255, thickness=line_thickness)  # type: ignore

        # Make the contour bold
        for i in range(0, len(x)):
            cv2.circle(img_copy, (x[i], y[i]), radius=1, color=255, thickness=-1)  # type: ignore
        img_copy[self.ground_height.value(), :] = 125

        # Add info signature
        crossSectionArea = cv2.contourArea(max(contour_object, key=cv2.contourArea))  # type: ignore
        signature(img_copy, [10, 20], 0.4, self.fps.value())
        printText(img_copy, [math.degrees(angle_left), math.degrees(angle_right),
                             x_right[np.argmax(y_right)] - x_left[np.argmax(y_left)],
                             crossSectionArea], [10, 60], 0.4)

        if ret:
            return img_copy, \
                   math.degrees(angle_left), \
                   math.degrees(angle_right), x_right[np.argmax(y_right)] - x_left[np.argmax(y_left)], \
                   crossSectionArea
        else:
            return img_copy

    def print_message(self, msg: str) -> None:
        current_time = datetime.now().strftime("%H:%M:%S")
        self.output_panel.append(f'{current_time}: {msg}')

    def create_gif(self):
        if os.path.exists(src.TEMP_DIR):
            frames = [Image.open(image) for image in glob.glob(f"{src.TEMP_DIR}/*.png")]  # type: ignore
        else:
            return
        frame_one = frames[0]
        path_to_save_gif = f'{self.project_directory}/{self.project_name}.gif'
        frame_one.save(str(path_to_save_gif), format="GIF", append_images=frames,
                       save_all=True, duration=100, loop=0)

    def set_post_widget(self):
        """ Set up interface for postprocessor QDockWidget. """

        self.data_gbox = QGroupBox("Data selection")  # noqa
        self.dock_wid_cont = QWidget()  # noqa
        self.m_layout = QVBoxLayout()  # noqa
        self.m_layout.setContentsMargins(5, 5, 5, 5)

        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.data_gbox.sizePolicy().hasHeightForWidth())
        self.data_gbox.setSizePolicy(sizePolicy)
        self.data_gbox.setMinimumSize(QSize(0, 100))
        self.data_gbox.setMaximumSize(QSize(16777215, 16777215))
        self.data_gbox.setFlat(False)
        self.data_gbox.setCheckable(False)
        self.data_gbox.setObjectName("data_gbox")
        self.m_grid_layout_1 = QGridLayout(self.data_gbox)  # noqa
        self.m_grid_layout_1.setContentsMargins(5, 5, 5, 5)
        self.m_grid_layout_1.setObjectName("m_grid_layout_1")

        font = QFont("Open Sans", 8)  # Create a QFont object with desired font properties
        self.data_gbox.setFont(font)
        self.dock_wid_cont.setFont(font)
        self.browse_csv = QPushButton(self.data_gbox)  # noqa
        self.browse_csv.setObjectName("browse_csv")
        self.browse_csv.setText('Browse')
        self.browse_csv.setFont(font)
        self.m_grid_layout_1.addWidget(self.browse_csv, 1, 1, 1, 1)
        spacerItem1 = QSpacerItem(210, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.m_grid_layout_1.addItem(spacerItem1, 1, 0, 1, 1)

        self.csv_list = QTreeView(self.data_gbox)  # noqa
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.csv_list.sizePolicy().hasHeightForWidth())
        self.csv_list.setSizePolicy(sizePolicy)
        self.data_gbox.setMinimumSize(QSize(0, 343))
        self.csv_list.setFrameShape(QFrame.Box)
        self.csv_list.setFrameShadow(QFrame.Sunken)
        self.csv_list.setLineWidth(1)
        self.csv_list.setObjectName("csv_list")
        self.m_grid_layout_1.addWidget(self.csv_list, 0, 0, 1, 2)

        self.csv_file_model = QFileSystemModel()  # noqa
        self.csv_file_model.setRootPath("")
        self.csv_list.setModel(self.csv_file_model)
        self.csv_list.setRootIndex(self.csv_file_model.index(""))
        self.csv_list.setFont(font)

        self.plot_options_gbox = plotControl(self.dock_wid_cont)  # noqa
        self.plot_options_gbox.setTitle('Plot controls')
        self.plot_options_gbox.setStyleSheet(src.QSS_Q_GROUP)
        self.data_gbox.setStyleSheet(src.QSS_Q_GROUP)

        self.fft_options_gbox = fftControl(self.dock_wid_cont)  # noqa
        self.fft_options_gbox.setTitle('FFT analysis')
        self.fft_options_gbox.setStyleSheet(src.QSS_Q_GROUP)
        self.post_tabWidget.setStyleSheet(src.QSS_Q_CHILD_TAB_WIDGET)

        splitter_top = QSplitter(Qt.Vertical)  # Create a splitter
        splitter_top.addWidget(self.data_gbox)
        splitter_top.addWidget(self.plot_options_gbox)
        splitter_top.addWidget(self.fft_options_gbox)
        self.m_layout.addWidget(splitter_top)

        self.dock_wid_cont.setLayout(self.m_layout)
        self.post_dock_widget.setWidget(self.dock_wid_cont)
        self.m_layout.setContentsMargins(0, 0, 0, 0)
        self.dock_wid_cont.setContentsMargins(2, 2, 2, 2)

    def plot(self, path: str) -> None:
        data = csvReader(path, self.plot_options_gbox.avg_scope.value())
        ax1, ax2, ax3 = self.pltWidget.axes

        # Clear data stored in axes
        for _ax in self.pltWidget.axes:
            _ax.clear()

        line_1, = ax1.plot(data.time, data.left_angle, label='Left angle', color='navy',
                           linewidth=self.plot_options_gbox.line_width.value())
        line_2, = ax2.plot(data.time, data.right_angle, label='Right angle', color='deepskyblue',
                           linewidth=self.plot_options_gbox.line_width.value())  # noqa

        line_3, = ax1.plot(data.time, [data.left_static_angle for x in data.left_angle],
                           color='navy', linestyle=(0, (5, 5)),
                           linewidth=self.plot_options_gbox.line_width.value(),
                           label=f'Static angle={data.left_static_angle}')  # noqa
        line_4, = ax2.plot(data.time, [data.right_static_angle for x in data.right_angle],
                           color='navy', linestyle=(0, (5, 5)),
                           linewidth=self.plot_options_gbox.line_width.value(),
                           label=f'Static angle={data.right_static_angle}')  # noqa

        line_5, = ax3.plot(
            data.time[0:len(data.dContact_length)], data.dContact_length,
            label='Length of droplet contact zone', color='deepskyblue', linewidth=0.5)  # noqa

        ax1.legend(handles=[line_1, line_3])
        ax2.legend(handles=[line_2, line_4])
        ax3.legend(handles=[line_5])

        ax1.grid(visible=True, which='both', linestyle='--', linewidth='0.25')
        ax2.grid(visible=True, which='both', linestyle='--', linewidth='0.25')
        ax3.grid(visible=True, which='both', linestyle='--', linewidth='0.25')

        ax1.set(xlabel="time [s]", ylabel="Angle [degrees]")
        ax2.set(xlabel="time [s]", ylabel="Angle [degrees]")
        ax3.set(xlabel="time [s]", ylabel="dL [pixels]")

        ax3.set_ylim([self.plot_options_gbox.dl_min.value(),
                      self.plot_options_gbox.dl_max.value()])
        ax1.set_title(self.plot_options_gbox.title_line.text())

    def on_plot_clicked(self) -> None:
        """ Plotting raw data in 'Raw data' tab. """
        first_index = self.csv_file_model.index(0, 0, self.csv_list.rootIndex())
        first_file_path = self.csv_file_model.filePath(first_index)

        data_path = first_file_path
        selected_item = None
        if self.csv_list.selectedIndexes():
            for index in self.csv_list.selectedIndexes():
                selected_item = self.csv_file_model.filePath(index)
            data_path = selected_item

        _, extension = os.path.splitext(data_path)
        if extension.lower() != '.csv':
            QMessageBox.warning(
                self, src.PROJ_NAME,
                'First you have to select directory containing CSV data.\n' +
                'Operation aborted.')
            return

        self.print_message(f'CSV file path: {data_path}')

        try:
            self.switch_to_tab("Raw data")
            self.plot(data_path)
            self.pltWidget.canvas.draw()
            self.print_message('INFO: Plot has been generated.')
        except Exception as e:
            QMessageBox.warning(
                self, src.PROJ_NAME,
                'Error while generating plot.\n' +
                f'{e}')
            self.print_message(str(e))
            return

    def plot_fft_clicked(self) -> None:
        first_index = self.csv_file_model.index(0, 0, self.csv_list.rootIndex())
        first_file_path = self.csv_file_model.filePath(first_index)

        data_path = first_file_path
        selected_item = None
        if self.csv_list.selectedIndexes():
            for index in self.csv_list.selectedIndexes():
                selected_item = self.csv_file_model.filePath(index)
            data_path = selected_item

        _, extension = os.path.splitext(data_path)
        if extension.lower() != '.csv':
            QMessageBox.warning(
                self, src.PROJ_NAME,
                'First you have to select directory containing CSV data.\n' +
                'Operation aborted.')
            return

        self.print_message(f'CSV file path: {data_path}')

        try:
            self.switch_to_tab("FFT Analysis")
            self.plot_fft(data_path)
            self.fftWidget.canvas.draw()
            self.print_message('INFO: FFT plot has been generated.')

        except Exception as e:
            self.print_message(str(e))
            QMessageBox.warning(
                self, src.PROJ_NAME,
                'Error while generating plot.\n' +
                f'{e}')
            return

    def plot_fft(self, path: str) -> None:
        # Fixme: Bug with plotting already existing data
        data = FFTAnalysis(path, self.plot_options_gbox.avg_scope.value())
        ax1, ax2 = self.fftWidget.axes

        if data.label in self._label_left:
            return

        if self.fft_options_gbox.common_graph.isChecked():
            if self._label_left and self._label_left[0] == 'Left Angle':
                self._label_left.pop(0)
                self._label_right.pop(0)

            line_1, = ax1.plot(data.frequencies[1:], data.left_fft_values[1:],
                                label=f'Left angle:{data.label}',
                                linewidth=self.fft_options_gbox.line_width.value())
            line_2, = ax2.plot(data.frequencies[1:], data.right_fft_values[1:],
                                label=f'Right angle: {data.label}',
                                linewidth=self.fft_options_gbox.line_width.value())  # noqa
            self._label_left.append(line_1)
            self._label_right.append(line_2)

        else:
            # Clear data stored in axes
            for _ax in self.fftWidget.axes:
                _ax.clear()

            if self._label_left:
                self._label_left.pop()
                self._label_right.pop()

            line_1, = ax1.plot(data.frequencies[1:], data.left_fft_values[1:], color='navy',
                               label='Left angle', linewidth=self.fft_options_gbox.line_width.value())
            line_2, = ax2.plot(data.frequencies[1:], data.right_fft_values[1:], color='navy',
                               label='Right angle', linewidth=self.fft_options_gbox.line_width.value())  # noqa
            self._label_left.append(line_1)
            self._label_right.append(line_2)

        ax1.legend(handles=self._label_left, fontsize=8)
        ax2.legend(handles=self._label_right, fontsize=8)

        ax1.grid(visible=True, which='both', linestyle='--', linewidth='0.25')
        ax2.grid(visible=True, which='both', linestyle='--', linewidth='0.25')

        ax1.set(xlabel="Frequency [Hz]", ylabel="Amplitude")
        ax2.set(xlabel="Frequency [Hz]", ylabel="Amplitude")

        ax1.set_xlim([
            self.fft_options_gbox.freq_min.value(),
            self.fft_options_gbox.freq_max.value()])
        ax2.set_xlim([
            self.fft_options_gbox.freq_min.value(),
            self.fft_options_gbox.freq_max.value()])

        ax1.set_title(self.fft_options_gbox.title_line.text())

    def switch_to_tab(self, tab_name: str) -> None:
        """ Find the index of the tab by its name. """
        if tab_name == "FFT Analysis":
            self.post_tabWidget.setCurrentIndex(1)
        else:
            self.post_tabWidget.setCurrentIndex(0)
        self.post_tabWidget.update()

    def on_clear_plot_clicked(self):
        self._label_left.clear()
        self._label_right.clear()
        try:
            # Clear data stored in axes
            for _ax in self.fftWidget.axes:
                _ax.clear()
            self.print_message('INFO: FFT plot cleared.')
        except Exception as e:
            self.print_message(str(e))
