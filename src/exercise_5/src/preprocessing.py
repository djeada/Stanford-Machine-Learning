""""
The goal of this module is to implement all readers and parser
needed to import the data for the Task 5 from the coding homeworks
in the Machine Learning course on coursera.com.
"""

from typing import Tuple, Iterable
import numpy as np
from pathlib import Path
import scipy.io


def read_data(
    path: Path,
    column_names: Tuple[str],
) -> Tuple[np.ndarray]:
    """
    Reads the data from a CSV file and returns the data as a numpy array.

    :param path: The path to the CSV file.
    :return: The data as a numpy array.
    """

    raw_data = scipy.io.loadmat(path)

    result = []
    for column_name in column_names:
        result.append(raw_data[column_name])

    return tuple(result)


def include_intercept(x_datasets: Iterable[np.ndarray]) -> Iterable[np.ndarray]:
    """
    Adds a column of ones to the given datasets.

    :param x_datasets: The datasets to add the column of ones to.
    :return: The datasets with the column of ones.
    """

    return [np.insert(x, 0, np.ones(len(x)), axis=1) for x in x_datasets]
