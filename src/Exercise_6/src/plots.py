""""
The goal of this module is to implement all the visualization
tools needed to graph the data and results of the computations
for the Task 6 from the coding homeworks in the Machine Learning
course on coursera.com.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_data(data: tuple, title: str = "scatter plot of training data 1") -> None:
    """
    Plots the data points x and y into a scatter plot. Two different colors are used 
    to distinguish the positive and negative examples.

    Args:
      data:
        A tuple of x and y values for the points to be plotted.

      title:
        A string that serves as both the plot's title and the saved figure's filename.

    Returns:
      None
    """
    pos, neg = data
    plt.figure(figsize=(10, 6))
    plt.scatter(pos[:, 0], pos[:, 1], marker="+", c="k", s=150, label="Positive Sample")
    plt.scatter(neg[:, 0], neg[:, 1], marker="o", c="y", s=100, label="Negative Sample")
    plt.xlabel("Column 1 Variable")
    plt.ylabel("Column 2 Variable")
    plt.legend()
    plt.title(title)
    plt.savefig(title.replace(" ", "_").lower())


def plot_boundary(data: tuple) -> None:
    """
    Boundary is drawn for the SVM classifier.

    Args:
      data: Contains the gaussian kernel and the extreme values of the data.
    
    Returns:
      None
    """
    clf, x_min, x_max = data
    w = clf.coef_[0]
    a = -w[0] / w[1]

    xp = np.array(np.linspace(x_min, x_max, 100))
    yp = a * xp - (clf.intercept_[0]) / w[1]
    plt.plot(xp, yp, "-k")


def plot_boundary_gaussian(data: tuple, n: int = 200) -> None:
    """
    Boundary is drawn for the SVM classifier with gaussian kernel.

    Args:
      data: Contains the gaussian kernel and the extreme values of the data.
      n: Number of points to be plotted.

    Returns:
      None
    """
    clf, x_min, x_max, y_min, y_max = data

    x_range = np.array(np.linspace(x_min, x_max, n))
    y_range = np.array(np.linspace(y_min, y_max, n))
    u, v = np.meshgrid(x_range, y_range)
    grid = np.array(list(zip(u.flatten(), v.flatten())))

    prediction_grid = clf.predict(grid).reshape((n, n))
    plt.contour(x_range, y_range, prediction_grid, cmap=plt.cm.coolwarm, extend="both")
