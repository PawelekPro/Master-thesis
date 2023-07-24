import argparse
import os.path
import sys
from typing import Tuple, Any

import pandas as pd
from matplotlib import pyplot as plt


class slipReader:
    time: list = []
    slip_data: list = []

    def __init__(self, dir_path: str):
        self.dir_path = dir_path
        self.time, self.slip_data = self.read_csv(self.dir_path)
        self.plot()

    @staticmethod
    def read_csv(file_path: str) -> Tuple[Any, Any]:
        df = pd.read_csv(file_path)

        time = df['Time'].to_numpy()
        dL = df['l0'].to_numpy()

        return time, dL

    def plot(self) -> None:
        plt.plot(self.time, self.slip_data)
        title = os.path.basename(self.dir_path)
        plt.title(title)
        plt.show()


def create_arg_parser():
    # Creates and returns the ArgumentParser object

    parser = argparse.ArgumentParser(description='Description of your app.')
    parser.add_argument('inputDirectory',
                    help='Path to the input directory.')

    return parser


if __name__ == "__main__":
    arg_parser = create_arg_parser()
    parsed_args = arg_parser.parse_args(sys.argv[1:])
    if os.path.exists(parsed_args.inputDirectory):
        print("Path: ", parsed_args.inputDirectory)
        testClass = slipReader(parsed_args.inputDirectory)
    else:
        print('Path does not exist.')
