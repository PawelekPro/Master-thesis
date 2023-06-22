import os

import numpy as np

from src.postprocessing.csvReader import csvReader
from scipy.fft import fft, fftfreq


class FFTAnalysis:
    def __init__(self, path: str, avg_scope: int, motion_freq: float):
        data = csvReader(path, avg_scope)
        self.label: str = os.path.basename(path)
        self.motion_freq = motion_freq

        sampling_rate = 1 / (data.time[1] - data.time[0])
        n = len(data.left_angle)
        self.frequencies = fftfreq(n, 1 / sampling_rate) / self.motion_freq

        # Perform the FFT
        self.left_fft_values = fft(data.left_angle) / len(data.left_angle)
        self.left_fft_values = np.abs(self.left_fft_values)

        self.right_fft_values = fft(data.right_angle) / len(data.right_angle)
        self.right_fft_values = np.abs(self.right_fft_values)
