""""
The goal of this module is to implement all readers and parser
needed to import the data for the Task 7 from the coding homeworks
in the Machine Learning course on coursera.com.
"""

from typing import Tuple
import numpy as np
from pathlib import Path
import scipy.io as sio
from PIL import Image


def read_data(path: Path) -> np.ndarray:
    """
    Reads the data from the path and returns the data as a numpy array.

    Args:
        path: The path to the data.
    
    Returns:
        The data as a numpy array.
    """

    data = sio.loadmat(str(path))
    x = data["X"]

    return x


def read_image(path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads the image from the path and returns the image as a numpy array.

    Args:
        path: The path to the image.

    Returns:
        The image as a numpy array and the shape of the image.
    """
    x = np.asarray(Image.open(path)) / 255
    original_shape = x.shape
    x = x.reshape(-1, 3)

    return x, original_shape
