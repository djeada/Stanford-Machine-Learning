""""
The goal of this module is to implement all readers and parser
needed to import the data for the Task 1 from the coding homeworks
in the Machine Learning course on coursera.com.
"""
from typing import Tuple

import numpy as np
from pathlib import Path


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


class FeatureNormalizer:
    """
    A class that normalizes features by subtracting the mean and dividing by the standard deviation.
    """

    def __init__(self, x: np.ndarray, excluded_columns: Tuple[int, ...] = ()):
        """
        Initializes a new instance of the FeatureNormalizer class.

        :param x: The input array.
        :param excluded_columns: The columns to exclude from normalization.
        """
        self.means = np.mean(x, axis=0)
        self.stds = np.std(x, axis=0)

        for i in excluded_columns:
            self.means[i] = 0
            self.stds[i] = 1

    def normalize(self, x: np.ndarray, epsilon=1e-100) -> np.ndarray:
        """
        Normalizes the input array.

        :param x: The input array.
        :param epsilon: A small value to avoid division by zero.
        :return: The normalized input array.
        """
        return (x - self.means) / (self.stds + epsilon)

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """
        Denormalizes the input array.

        :param x: The input array.
        :return: The denormalized input array.
        """
        return x * self.stds + self.means
