""""
The goal of this module is to implement all readers and parser
needed to import the data for the Task 7 from the coding homeworks
in the Machine Learning course on coursera.com.
"""

import numpy as np
from pathlib import Path
import scipy.io as sio
from PIL import Image


def read_data(path: Path):
    """
    x is a matrix with m rows and n columns
    """

    data = sio.loadmat(str(path))
    x = data["X"]

    return x


def read_image(path):
    x = np.asarray(Image.open(path)) / 255
    original_shape = x.shape
    x = x.reshape(-1, 3)

    return x, original_shape
