""""
The goal of this module is to implement all readers and parser
needed to import the data for the Task 3 from the coding homeworks
in the Machine Learning course on coursera.com.
"""
from typing import Tuple

import scipy.io
import numpy as np
from pathlib import Path


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

    return data["Theta1"], data["Theta2"]
