""""
The goal of this module is to implement all readers and parser
needed to import the data for the Task 4 from the coding homeworks
in the Machine Learning course on coursera.com.
"""
from typing import Tuple

import numpy as np
from pathlib import Path
import scipy.io


def read_data(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read the data consisting of:
    - An X matrix with m rows and n columns containing 5000 handwritten digit training samples.
    - an m-element vector y containing the labels for the corresponding digits

    Args:
      path:
        The input file's path.

    Returns:
      A tuple consisting of X matrix and y vector.
    """

    data = scipy.io.loadmat(str(path))
    x, y = data["X"], data["y"]
    ones = np.ones(len(x))
    # prepend column of ones
    x = np.insert(x, 0, ones, axis=1)

    return x, y


def read_weights(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read the weights used in the neural network.

    Args:
      path:
        The input file's path.

    Returns:
      A tuple consisting of two sets of weights.
    """
    data = scipy.io.loadmat(str(path))

    theta_1, theta_2 = data["Theta1"], data["Theta2"]
    theta_2 = np.roll(theta_2, 1, axis=0)

    return theta_1, theta_2


def clean_y(y: np.ndarray) -> None:
    """
    Originally, 10 represented 0. Let's use 0 for 0.

     Args:
      y:
        An array with digit labels.

    Returns:
        None.
    """

    y = y.ravel()

    for i, elem in enumerate(y):
        if elem == 10:
            y[i] = 0
