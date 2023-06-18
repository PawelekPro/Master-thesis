import os

import numpy as np

from src.postprocessing.csvReader import csvReader
from scipy.fft import fft, fftfreq


class FFTAnalysis:
    def __init__(self, path: str, avg_scope: int):
        data = csvReader(path, avg_scope)
        self.label: str = os.path.basename(path)

        sampling_rate = 1 / (data.time[1] - data.time[0])
        n = len(data.left_angle)
        self.frequencies = fftfreq(n, 1 / sampling_rate)

        # Perform the FFT
        self.left_fft_values = fft(data.left_angle) / len(data.left_angle)
        self.left_fft_values = np.abs(self.left_fft_values)

        self.right_fft_values = fft(data.right_angle) / len(data.right_angle)
        self.right_fft_values = np.abs(self.right_fft_values)
