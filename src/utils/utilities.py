from typing import Tuple

import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QPainterPath, QPixmap, QBrush, QPolygon, QImage
from PyQt5.QtWidgets import QLabel
from datetime import datetime

MEAS_DATE = datetime.now().date()


def create_rounded_pixmap(image, radius):
    # Convert the OpenCV image to a QImage
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image_rgb.shape
    qimage = QImage(image_rgb.data, width, height, QImage.Format_RGB888)

    # Create a QPixmap and scale it to the desired size
    pixmap = QPixmap.fromImage(qimage).scaled(width, height, Qt.KeepAspectRatioByExpanding)

    # Create a transparent pixmap to draw the rounded image on
    rounded_pixmap = QPixmap(width, height)
    rounded_pixmap.fill(Qt.transparent)

    # Create a QPainter and set antialiasing
    painter = QPainter(rounded_pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)

    # Create a QPainterPath for the rounded rectangle
    path = QPainterPath()
    path.addRoundedRect(0, 0, width, height, radius, radius)

    # Set the QPainter's clip region to the rounded rectangle
    painter.setClipPath(path)

    # Create a QBrush with the pixmap as the texture
    brush = QBrush(pixmap)

    # Set the brush on the QPainter
    painter.setBrush(brush)

    # Draw the rounded rectangle with the custom image
    painter.drawRoundedRect(0, 0, width, height, radius, radius)

    painter.end()

    return rounded_pixmap


def printText(image, value, org, fontScale):
    font = cv2.FONT_HERSHEY_SIMPLEX

    color = 255
    # Line thickness of 1 px
    thickness = 1
    image = cv2.putText(image, "Measured data:", (org[0], org[1]), font,
                        fontScale, color, thickness, cv2.LINE_AA)

    image = cv2.putText(image, "Left angle [deg]: %.2f" % value[0], (org[0], org[1] + 20), font,
                        fontScale, color, thickness, cv2.LINE_AA)

    image = cv2.putText(image, "Right angle [deg]: %.2f" % value[1], (org[0], org[1] + 40), font,
                        fontScale, color, thickness, cv2.LINE_AA)

    image = cv2.putText(image, "Contact length [px]: %.2f" % value[2], (org[0], org[1] + 60), font,
                        fontScale, color, thickness, cv2.LINE_AA)

    image = cv2.putText(image, "Cross section area [px2]: %.2f" % value[3], (org[0], org[1] + 80), font,
                        fontScale, color, thickness, cv2.LINE_AA)
    return image


def signature(image, origin, font_scale, fps):
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = 255
    # Line thickness of 1 px
    thickness = 1

    image = cv2.putText(image, f"Measurement date: {MEAS_DATE}", (origin[0], origin[1]), font,
                        font_scale, color, thickness, cv2.LINE_AA)
    image = cv2.putText(image, f"Frame rate: {fps} fps", (origin[0], origin[1] + 20), font,
                        font_scale, color, thickness, cv2.LINE_AA)
    return image


def drawCSYS(image, org, thickness, scale, color, optional_text=""):
    start_point = (org[0], org[1])
    x_end_point = (org[0], org[1] - int(scale * 50))
    y_end_point = (org[0] + int(scale * 50), org[1])
    image = cv2.arrowedLine(image, start_point, x_end_point, color, thickness, tipLength=0.2)
    image = cv2.arrowedLine(image, start_point, y_end_point, color, thickness, tipLength=0.2)

    if len(optional_text) != 0:
        image = cv2.putText(image, "%s" % optional_text,
                            (org[0] - int(scale * 25), org[1] + int(scale * 15)), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.4,
                            0, 1, cv2.LINE_AA)
    return image


def draw_ruler(image, length: int, numb_of_div: int, orig_coord: Tuple[int, int]) -> None:
    cv2.line(image, orig_coord,                                          # type: ignore
             (orig_coord[0], orig_coord[1] - length), 230, thickness=1)

    for i in range(numb_of_div + 1):
        cv2.line(image, (orig_coord[0], orig_coord[1] - int(i * length/numb_of_div)),  # type: ignore
                 (orig_coord[0] + 7, orig_coord[1] - int(i * length/numb_of_div)), 210, thickness=1)