""""
The goal of this module is to implement all readers and parser
needed to import the data for the Task 8 from the coding homeworks
in the Machine Learning course on coursera.com.
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import scipy.io as sio


def read_data(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads the data from the path and returns the data as a tuple.

    Args:
        path: Path to the data.

    Returns:
        A tuple containing the data.
    """

    data = sio.loadmat(str(path))
    x = data["X"]
    x_val = data["Xval"]
    y_val = data["yval"]
    return x, x_val, y_val


def read_movie_data(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads the movie data from the path and returns the data as a tuple.

    Args:
        path: Path to the data.

    Returns:
        A tuple containing the data.
    """
    data = sio.loadmat(str(path))
    y = data["Y"]
    r = data["R"]

    return y, r


def read_movie_params(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads the movie parameters from the path and returns the data as a tuple.

    Args:
        path: Path to the data.

    Returns:
        A tuple containing the data.
    """
    data = sio.loadmat(f"{path}")
    x = data["X"]
    theta = data["Theta"]

    return x, theta


def read_movie_ids(path: Path) -> list[str]:
    """
    Reads the movie ids from the path and returns the data as a numpy array.

    Args:
        path: Path to the data.

    Returns:
        A list of strings representing movie ids.
    """
    with open(path, encoding="ascii", errors="surrogateescape") as _file:
        movies = [" ".join(line.strip("\n").split(" ")[1:]) for line in _file]

    return movies
