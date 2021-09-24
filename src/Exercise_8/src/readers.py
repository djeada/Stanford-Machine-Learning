""""
The goal of this module is to implement all readers and parser
needed to import the data for the Task 8 from the coding homeworks
in the Machine Learning course on coursera.com.
"""

from pathlib import Path
import scipy.io as sio


def read_data(path: Path) -> tuple:
    """
    x is a matrix with m rows and n columns
    """

    data = sio.loadmat(str(path))
    x = data["X"]
    x_val = data["Xval"]
    y_val = data["yval"]
    return x, x_val, y_val


def read_movie_data(path) -> tuple:
    """
    """
    data = sio.loadmat(str(path))
    y = data["Y"]
    r = data["R"]

    return y, r


def read_movie_params(path) -> tuple:
    data = sio.loadmat(path)
    x = data["X"]
    theta = data["Theta"]

    return x, theta


def read_movie_ids(path: Path):
    with open(path, encoding="ascii", errors="surrogateescape") as _file:
        movies = [" ".join(line.strip("\n").split(" ")[1:]) for line in _file]

    return movies
