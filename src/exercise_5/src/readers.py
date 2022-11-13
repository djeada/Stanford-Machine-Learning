""""
The goal of this module is to implement all readers and parser
needed to import the data for the Task 5 from the coding homeworks
in the Machine Learning course on coursera.com.
"""

from typing import Tuple
import numpy as np
from pathlib import Path
import scipy.io


def read_data(
    path: Path
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """
    Reads the data from the given path and returns the training, validation and test data.

    Args:
        path: The path to the data.
    
    Returns:
        A tuple containing the training, validation and test data.
    """

    raw_data = scipy.io.loadmat(f"{path}")
    x, y = raw_data["X"], raw_data["y"]
    x_validation, y_validation = raw_data["Xval"], raw_data["yval"]
    x_test, y_test = raw_data["Xtest"], raw_data["ytest"]

    x = np.insert(x, 0, np.ones(len(x)), axis=1)
    x_validation = np.insert(x_validation, 0, np.ones(len(x_validation)), axis=1)
    x_test = np.insert(x_test, 0, np.ones(len(x_test)), axis=1)

    return (x, y), (x_validation, y_validation), (x_test, y_test)
