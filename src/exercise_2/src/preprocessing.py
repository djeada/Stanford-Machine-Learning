""""
The goal of this module is to implement all readers and parser
needed to import the data for the Task 2 from the coding homeworks
in the Machine Learning course on coursera.com.
"""

from typing import Tuple

import numpy as np
from pathlib import Path
from dataclasses import dataclass


def read_data(path: Path) -> np.ndarray:
    """
    Reads the data from a CSV file and returns the data as a numpy array.

    :param path: The path to the CSV file.
    :return: The data as a numpy array.
    """

    data = np.loadtxt(str(path), delimiter=",", unpack=True)
    return data.T


def split_input_output(
    data: np.ndarray, input_columns: Tuple[int, ...], output_columns: Tuple[int, ...]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits the data into input and output arrays.

    :param data: The data to split.
    :param input_columns: The columns to use as input.
    :param output_columns: The columns to use as output.
    :return: A tuple containing the input and output arrays.
    """

    x = data[:, input_columns]
    y = data[:, output_columns]

    # flatten if only one column is used for input or output
    if len(input_columns) == 1:
        x = x.flatten()
    if len(output_columns) == 1:
        y = y.flatten()

    return x, y


def include_intercept(x: np.ndarray) -> np.ndarray:
    """
    Prepends a column of ones to the input array.

    :param x: The input array.
    :return: The input array with a column of ones prepended.
    """

    if x.ndim == 1:
        x = x[:, np.newaxis]

    return np.insert(x, 0, 1, axis=1)


@dataclass
class Data:
    positive: np.ndarray
    negative: np.ndarray

    @classmethod
    def from_data(cls, x: np.ndarray, y: np.ndarray) -> "Data":
        """
        Creates a Data object from the input and output arrays.

        :param x: The input array.
        :param y: The output array.
        :return: A Data object.
        """

        positive = x[y == 1]
        negative = x[y == 0]

        return cls(positive, negative)


class FeatureMapper:
    """
    Maps the features of the input array to polynomial features.
    """

    def __init__(self, n_degrees: int = 6):
        self.n_degrees = n_degrees

    def map(self, x1_col: np.ndarray, x2_col: np.ndarray) -> np.ndarray:
        """
        Maps the features of the input array to polynomial features.

        :param x1_col: The first feature column.
        :param x2_col: The second feature column.
        :return: The mapped features.
        """

        result = np.ones((x1_col.shape[0], 1))

        for i in range(self.n_degrees):
            for j in range(i + 2):
                a = x1_col ** (i - j + 1)
                b = x2_col ** j
                result = np.hstack((result, (a * b).reshape(a.shape[0], 1)))

        return result
