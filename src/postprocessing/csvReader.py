from typing import Tuple, Any

import pandas as pd


class csvReader:
    def __init__(self, path: str, avg_scope: int):
        self.path = path
        self.avg_scope = avg_scope

        self.time, self.left_angle, self.right_angle, self.contact_length = self.read_csv()
        self.avg_contact_length = self.moving_average(self.contact_length)

        left_static_angle = 0
        right_static_angle = 0
        contact_static_length = 0

        for i in range(50):
            left_static_angle += self.left_angle[i]
            right_static_angle += self.right_angle[i]
            contact_static_length += self.contact_length[i]

        self.left_static_angle = round(left_static_angle / 50, 2)
        self.right_static_angle = round(right_static_angle / 50, 2)
        self.contact_static_length = round(contact_static_length / 50, 2)

        self.dContact_length = [_len - self.contact_static_length for _len in self.avg_contact_length]

    def read_csv(self) -> Tuple[Any, Any, Any, Any]:
        df = pd.read_csv(self.path)
        time = df['Time'].to_list()
        leftAngle = df['Left Angle'].to_list()
        rightAngle = df['Right Angle'].to_list()
        contactLength = df['Contact Length'].to_list()

        return time, leftAngle, rightAngle, contactLength

    def moving_average(self, data: list) -> list:
        avg_data = []
        iterator = 0
        while iterator < len(data) - self.avg_scope:
            buffor = data[iterator: iterator + self.avg_scope]  # noqa
            buffor_avarage = round(sum(buffor) / self.avg_scope, 2)  # noqa
            avg_data.append(buffor_avarage)
            iterator += 1
        return avg_data
