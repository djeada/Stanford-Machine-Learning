""""
The goal of this module is to implement all readers and parser
needed to import the data for the Task 3 from the coding homeworks
in the Machine Learning course on coursera.com.
"""

import scipy.io
import numpy as np
from pathlib import Path


def read_data(path: Path) -> tuple:
    """
    x is a matrix with m rows and n columns
    y is a matrix with m rows and 1 column
    """

    data = scipy.io.loadmat(str(path))
    x, y = data["X"], data["y"]
    ones = np.ones(len(x))
    # prepend column of ones
    x = np.insert(x, 0, ones, axis=1)

    return x, y


def read_weights(path: Path) -> tuple:
    data = scipy.io.loadmat(str(path))

    return data["Theta1"], data["Theta2"]
