""""
The goal of this module is to implement all readers and parser
needed to import the data for the Task 8 from the coding homeworks
in the Machine Learning course on coursera.com.
"""

from pathlib import Path
from typing import Tuple, List
import numpy as np
import scipy.io as sio


def read_data(
    path: Path, column_names: Tuple[str, ...] = ("X",)
) -> Tuple[np.ndarray, ...]:
    """
    Reads the data from the file.

    :param path: The path to the file.
    :param column_names: The names of the columns.
    :return: The data.
    """
    data = sio.loadmat(str(path))
    return tuple(data[column_name] for column_name in column_names)


def read_text_file(path: Path) -> List[str]:
    """
    Reads the text file.

    :param path: The path to the file.
    :return: The lines of the file as a list of strings.
    """
    content = path.read_text(encoding="ascii", errors="surrogateescape")
    return content.splitlines()


def strip_ids(ids: List[str]) -> List[str]:
    """
    Strips the ids from the lines.

    :param ids: The list of ids.
    :return: The list of ids without the prefix.
    """
    return [" ".join(id_.split(" ")[1:]) for id_ in ids]
