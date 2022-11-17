import numpy as np


def select_n_random_rows(x: np.ndarray, n: int = 20) -> np.ndarray:
    """ "
    Selects n random rows from x.

    :param x: The input array.
    :param n: The number of rows to select.
    :return: The indices of the selected rows.
    """

    random_indices = np.random.choice(x.shape[0], n)
    return random_indices
