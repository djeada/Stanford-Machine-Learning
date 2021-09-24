""""
The goal of this module is to implement all the visualization
tools needed to graph the data and results of the computations
for the Task 4 from the coding homeworks in the Machine Learning
course on coursera.com.
"""

import numpy as np
import matplotlib.pyplot as plt


def display_random_grid(x: np.ndarray, n: int = 20, indices: np.ndarray = None) -> None:
    if indices is None:
        indices = np.random.choice(x.shape[0], n)

    plt.figure(figsize=(6, 6))
    image = x[indices, 1:].reshape(-1, n).T
    plt.imshow(image)
    plt.axis("off")
