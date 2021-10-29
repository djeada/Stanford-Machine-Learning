""""
The goal of this module is to implement all the visualization
tools needed to graph the data and results of the computations
for the Task 4 from the coding homeworks in the Machine Learning
course on coursera.com.
"""

import numpy as np
import matplotlib.pyplot as plt


def display_random_grid(x: np.ndarray, n: int = 20, indices: np.ndarray = None) -> None:
    """"
    Display a grid with n digits on it. If no indices are specified,
    a grid of n random digits is displayed.

    Args:
      x:
       An array containing 5000 images. Each image is a row. Each image contains 400 pixels (20x20).
      n:
        Number of digits to be displayed.
      indices:
        The indices of the digits in matrix x.

    Returns:
      None
    """
    if indices is None:
        indices = np.random.choice(x.shape[0], n)

    plt.figure(figsize=(6, 6))
    image = x[indices, 1:].reshape(-1, n).T
    plt.imshow(image)
    plt.axis("off")
