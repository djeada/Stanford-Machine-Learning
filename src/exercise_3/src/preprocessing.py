""""
The goal of this module is to implement all readers and parser
needed to import the data for the Task 3 from the coding homeworks
in the Machine Learning course on coursera.com.
"""
from typing import Tuple

import scipy.io
import numpy as np
from pathlib import Path


def read_data(
    path: Path, input_label: str = "X", output_label: str = "y"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads the data from a CSV file and returns the data as a tuple of
    input and output arrays.

    :param path: The path to the CSV file.
    :param input_label: The label of the input column.
    :param output_label: The label of the output column.
    :return: The data as two numpy arrays.
    """

    data = scipy.io.loadmat(str(path))
    return data[input_label], data[output_label]


def include_intercept(x: np.ndarray) -> np.ndarray:
    """
    Prepends a column of ones to the input array.

    :param x: The input array.
    :return: The input array with a column of ones prepended.
    """

    if x.ndim == 1:
        x = x[:, np.newaxis]

    return np.insert(x, 0, 1, axis=1)


def map_labels(y: np.ndarray) -> np.ndarray:
    """
    Maps the labels to a range from 0 to 9.

    :param y: The labels.
    :return: The mapped labels.
    """
    return np.where(y == 10, 0, y)


def read_weights(path: Path, column_names: Tuple[str, ...]) -> Tuple[np.ndarray, ...]:
    """
    Reads the weights from a CSV file and returns them as a tuple of
    numpy arrays.

    :param path: The path to the CSV file.
    :param column_names: The names of the columns containing the weights.
    :return: The weights as a tuple of numpy arrays.
    """
    data = scipy.io.loadmat(str(path))

    result = []
    for column_name in column_names:
        if column_name not in data:
            raise ValueError(f"Column {column_name} not found in {path}")

        result.append(data[column_name])

    return tuple(result)
