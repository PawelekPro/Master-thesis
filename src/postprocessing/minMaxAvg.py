""" Utility which is not included in GUI yet. """
from pathlib import Path
from typing import Tuple, Any, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import ndarray

plt.style.use('seaborn-v0_8-dark-palette')


def get_sub(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    sub_s = "ₐ₈CDₑբGₕᵢⱼₖₗₘₙₒₚQᵣₛₜᵤᵥwₓᵧZₐ♭꜀ᑯₑբ₉ₕᵢⱼₖₗₘₙₒₚ૧ᵣₛₜᵤᵥwₓᵧ₂₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
    res = x.maketrans(''.join(normal), ''.join(sub_s))
    return x.translate(res)


class minMaxAvgManager:
    leftMin: list = []
    rightMin: list = []
    leftMax: list = []
    rightMax: list = []
    avgLeft: list = []
    avgRight: list = []
    contactLen: list = []
    leftStat: list = []
    rightStat: list = []
    props_list = [leftMax, rightMax, leftMin, rightMin, avgLeft,
                  avgRight, contactLen, leftStat, rightStat]

    def __init__(self, dir_path: str):
        self.dir_path = dir_path

        for path in Path(self.dir_path).glob("*.csv"):
            data = self.calculateData(str(path))
            for prop in self.props_list:
                prop.append(data[self.props_list.index(prop)])

    @staticmethod
    def read_csv(file_path: str) -> Tuple[Any, Any, Any, Any, Any, Any, Any]:
        df = pd.read_csv(file_path)

        time = df['Time'].to_numpy()
        leftAngle = df['Left Angle'].to_numpy()
        rightAngle = df['Right Angle'].to_numpy()
        contactLength = df['Contact Length'].to_numpy()
        crossSectionArea = df['Cross Section Area'].to_numpy()

        time_threshold = 2  # Value for which there is already steady state
        idx = np.where(np.asarray(time) > time_threshold)

        return time[idx], leftAngle[idx], rightAngle[idx], contactLength, crossSectionArea, leftAngle, rightAngle

    def calculateData(self, file_path: str) -> Tuple[Union[float, Any], Union[float, Any], Union[float, Any], Union[
            float, Any], ndarray, ndarray, ndarray, ndarray, ndarray]:
        time, leftAngle, rightAngle, contactLength, crossSectionArea, left, right = self.read_csv(file_path)

        no_divisions = 10
        div = int(np.size(time) / no_divisions)

        avgLeftMax = []
        for i in range(len(time)):
            if i % div == 0 and i > 0:
                avgLeftMax.append(max(leftAngle[i - div:i]))

        avgRightMax = []
        for i in range(len(time)):
            if i % div == 0 and i > 0:
                avgRightMax.append(max(rightAngle[i - div:i]))

        avgLeftMin = []
        for i in range(len(time)):
            if i % div == 0 and i > 0:
                avgLeftMin.append(min(leftAngle[i - div:i]))

        avgRightMin = []
        for i in range(len(time)):
            if i % div == 0 and i > 0:
                avgRightMin.append(min(rightAngle[i - div:i]))

        static_range = 50
        staticLength = np.mean(contactLength[0:static_range])
        leftStat = np.mean(left[0:static_range])
        rightStat = np.mean(right[0:static_range])

        # DEBUG
        # print("Max left:", sum(avgLeftMax) / len(avgLeftMax))
        # print("Max right:", sum(avgRightMax) / len(avgRightMax))
        # print("Min left:", sum(avgLeftMin) / len(avgLeftMin))
        # print("Min right:", sum(avgRightMin) / len(avgRightMin))
        # print("Mean left:", np.mean(leftAngle))
        # print("Mean right:", np.mean(rightAngle))
        # print("Static length:", staticLength)

        return sum(avgLeftMax) / len(avgLeftMax), sum(avgRightMax) / len(avgRightMax), \
            sum(avgLeftMin) / len(avgLeftMin), sum(avgRightMin) / len(avgRightMin), \
            np.mean(leftAngle), np.mean(rightAngle), staticLength, leftStat, rightStat

    def plot(self):
        plt.subplot(1, 2, 1)
        plt.errorbar([x/np.mean(self.contactLen) for x in self.contactLen], self.avgLeft,
                     [[i - j for i, j in zip(self.avgLeft, self.leftMin)],
                      [j - i for i, j in zip(self.avgLeft, self.leftMax)]],
                     fmt='o', linewidth=0.5, capsize=6)
        plt.title("Left angle")
        plt.xlabel("Normalised contact length (d_i/mean(D) where D = [d_1, d_2, ...]", size=10)
        plt.ylabel('[min(\u03F4{}), avg(\u03F4{}), max(\u03F4{})]'.format(get_sub('L'), get_sub('L'), get_sub('L')),
                   size=11)
        plt.grid(visible=True, which='both', linestyle='--', linewidth='0.25')
        for a, b in zip([x/np.mean(self.contactLen) for x in self.contactLen], self.avgLeft):
            plt.text(a, b, str('  ' + "{:.2f}".format(b)), size=7)
        for a, b in zip([x/np.mean(self.contactLen) for x in self.contactLen], self.leftMin):
            plt.text(a, b, str('  ' + "{:.2f}".format(b)), size=7)
        for a, b in zip([x/np.mean(self.contactLen) for x in self.contactLen], self.leftMax):
            plt.text(a, b, str('  ' + "{:.2f}".format(b)), size=7)

        # plt.ylim([48, 90])
        for i in range(len(self.contactLen)):
            plt.text([x/np.mean(self.contactLen) for x in self.contactLen][i], self.leftMax[i] + 1,
                     f'{round(self.leftStat[i], 2)}',
                     bbox=dict(facecolor='red', alpha=0.5, pad=1), size=6)

        plt.subplot(1, 2, 2)
        plt.errorbar([x/np.mean(self.contactLen) for x in self.contactLen], self.avgRight,
                     [[i - j for i, j in zip(self.avgRight, self.rightMin)],
                      [j - i for i, j in zip(self.avgRight, self.rightMax)]],
                     fmt='o', linewidth=0.5, capsize=6)
        plt.title("Right angle")
        plt.xlabel("Normalised contact length (d_i/mean(D) where D = [d_1, d_2, ...]", size=10)
        plt.ylabel('[min(\u03F4{}), avg(\u03F4{}), max(\u03F4{})]'.format(get_sub('P'), get_sub('P'), get_sub('P')),
                   size=11)
        plt.grid(visible=True, which='both', linestyle='--', linewidth='0.25')
        for a, b in zip([x/np.mean(self.contactLen) for x in self.contactLen], self.avgRight):
            plt.text(a, b, str('  ' + "{:.2f}".format(b)), size=7)
        for a, b in zip([x/np.mean(self.contactLen) for x in self.contactLen], self.rightMin):
            plt.text(a, b, str('  ' + "{:.2f}".format(b)), size=7)
        for a, b in zip([x/np.mean(self.contactLen) for x in self.contactLen], self.rightMax):
            plt.text(a, b, str('  ' + "{:.2f}".format(b)), size=7)

        # plt.ylim([48, 90])
        for i in range(len(self.contactLen)):
            plt.text([x/np.mean(self.contactLen) for x in self.contactLen][i], self.rightMax[i] + 1, f'{round(self.rightStat[i], 2)}',
                     bbox=dict(facecolor='red', alpha=0.5, pad=1), size=6)

        plt.suptitle(
            '[min(\u03F4), avg(\u03F4), max(\u03F4)](contact length)' + ' - zestawienie 20 pomiarów dla pleksy\n'
            'Czerwone boxy zawieraja wartość kąta statycznego', size=14)
        plt.show()


if __name__ == "__main__":
    testClass = minMaxAvgManager('C:/MEIL_WORKSPACE/')
    testClass.plot()
