""""
The goal of this module is to implement all readers and parser
needed to import the data for the Task 1 from the coding homeworks
in the Machine Learning course on coursera.com.
"""

import numpy as np
from pathlib import Path


def read_data(path: Path, use_cols: tuple = (0, 1)) -> tuple:
    """
    x is a matrix with m rows and n columns
    y is a matrix with m rows and 1 column
    return: None.
    """

    raw_data = np.loadtxt(str(path), delimiter=",", usecols=use_cols, unpack=True)
    x = np.transpose(np.array(raw_data[:-1]))
    ones = np.ones(len(x))
    # prepend column of ones
    x = np.insert(x, 0, ones, axis=1)
    y = np.transpose(np.array(raw_data[-1:]))

    return x, y
