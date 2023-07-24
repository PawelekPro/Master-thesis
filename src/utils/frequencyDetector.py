""" MODULE FOR CALCULATING DOMINANT FREQUENCY FROM OSCILLOSCOPE DATA
    DATA HAS TO BE SAVED IN SUCH FORMAT:
    TIME, ACTUAL_POSITION, DEMAND POSITION, ACTUAL VELOCITY, DEMAND VELOCITY
"""

import os
from typing import Tuple, Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class LinMotCsvDataExtractor:
    def __init__(self, path: str):
        self.path = path

        self.time, self.actualPos, self.demandPos, self. actualVel, self.demandVel = self.read_csv()

    def read_csv(self) -> Tuple[Any, Any, Any, Any, Any]:
        df = pd.read_csv(self.path)
        time = df['Time(s)'].to_list()
        actualPos = df['MC SW Overview - Actual Position(mm)'].to_list()
        demandPos = df['MC SW Overview - Demand Position(mm)'].to_list()
        actualVel = df['MC SW Overview - Actual Velocity(m/s)'].to_list()
        demandVel = df['MC SW Overview - Demand Velocity(m/s)'].to_list()

        return time, actualPos, demandPos, actualVel, demandVel


if __name__ == "__main__":
    path = 'D:/praca_magisterska/pomiary_24072023/osci/A100_F71.csv'

    data = None
    if os.path.isfile(path):
        data = LinMotCsvDataExtractor(path)

    if data is not None:
        plt.plot(data.time, data.demandPos, label='Demand position [mm]')
        plt.plot(data.time, data.actualVel, label='Actual position [mm]')
        plt.legend()
        plt.show()

    fft_result = np.fft.fft(data.actualPos)

    sampling_rate = 1 / (data.time[1] - data.time[0])
    N = len(data.time)

    freq = np.fft.fftfreq(N, d=1/sampling_rate)
    freq = freq[:N // 2]

    max_index = np.argmax(np.abs(fft_result[:N // 2]))
    dominant_frequency = freq[max_index]

    print("Dominant frequency:", dominant_frequency)