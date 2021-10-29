""""
The goal of this module is to implement all readers and parser
needed to import the data for the Task 2 from the coding homeworks
in the Machine Learning course on coursera.com.
"""

import numpy as np
from pathlib import Path


def read_data(path: Path, use_cols: tuple = (0, 1, 2)) -> tuple:
    """
    Reads the data from the given path and returns the X and y arrays. 
    The X array is a 2D array with the first column as the index and the second column as the value. 
    The y array is a 1D array with the labels.

    Args:
        path: The path to the data file.
        use_cols: The columns to use from the data file.

    Returns:
        A tuple consisting of:
        - x: The X array.
        - y: The y array.
    """

    raw_data = np.loadtxt(path, delimiter=",", usecols=use_cols, unpack=True)
    x = np.transpose(np.array(raw_data[:-1]))
    ones = np.ones(len(x))
    # prepend column of ones
    x = np.insert(x, 0, ones, axis=1)
    y = np.transpose(np.array(raw_data[-1:]))

    return x, y
