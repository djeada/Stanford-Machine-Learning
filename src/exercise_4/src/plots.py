""""
The goal of this module is to implement all the visualization
tools needed to graph the data and results of the computations
for the Task 4 from the coding homeworks in the Machine Learning
course on coursera.com.
"""

import numpy as np
import matplotlib.pyplot as plt


def display_grid_of_rows(
    x: np.ndarray, indices: np.ndarray, cell_size: int = 20, cells_per_row: int = 8
) -> None:
    """ "
    Displays a grid with n digits on it. Each index in the indices
    represents a single digit. There are maximal 10 digits displayed
    in a single row.

    :param x: The input array.
    :param indices: The indices of the digits in matrix x.
    :param cell_size: The size of each cell in the grid.
    :param cells_per_row: The number of cells per row.
    :return: None
    """
    # number of cells in a single column
    cells_per_column = int(np.ceil(len(indices) / cells_per_row))

    # create a new figure
    fig = plt.figure(figsize=(cells_per_row, cells_per_column))

    # iterate over all indices
    for i, index in enumerate(indices):
        # create a new subplot
        ax = fig.add_subplot(cells_per_column, cells_per_row, i + 1)
        # display the digit
        ax.imshow(x[index].reshape(cell_size, cell_size).T, cmap="gray")
        # remove the axis
        ax.axis("off")

    # show the figure
    plt.show()
