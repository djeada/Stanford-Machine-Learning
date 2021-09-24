""""
The goal of this module is to implement all readers and parser
needed to import the data for the Task 4 from the coding homeworks
in the Machine Learning course on coursera.com.
"""

import numpy as np
from pathlib import Path
import scipy.io


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

    theta_1, theta_2 = data["Theta1"], data["Theta2"]
    theta_2 = np.roll(theta_2, 1, axis=0)

    return theta_1, theta_2


def clean_y(y: np.ndarray) -> None:
    """
    Originally, 10 represented 0. Let's use 0 for 0.
    return: None.
    """

    y = y.ravel()

    for i, elem in enumerate(y):
        if elem == 10:
            y[i] = 0
