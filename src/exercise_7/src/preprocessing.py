""""
The goal of this module is to implement all readers and parser
needed to import the data for the Task 7 from the coding homeworks
in the Machine Learning course on coursera.com.
"""

from typing import Tuple
import numpy as np
from pathlib import Path
import scipy.io as sio
from PIL import Image


def read_data(
    path: Path, column_names: Tuple[str, ...] = ("X",)
) -> Tuple[np.ndarray, ...]:
    """
    Reads the data from the file.

    :param path: The path to the file.
    :param column_names: The names of the columns.
    :return: The data.
    """
    data = sio.loadmat(str(path))
    return tuple(data[column_name] for column_name in column_names)


def read_image(path: Path) -> np.ndarray:
    """
    Reads the image from the file.

    :param path: The path to the file.
    :return: The image.
    """
    x = np.asarray(Image.open(path)) / 255
    original_shape = x.shape
    x = x.reshape(-1, 3)

    return x, original_shape


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
