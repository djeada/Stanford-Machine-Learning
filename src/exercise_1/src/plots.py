""""
The goal of this module is to implement all the visualization
tools needed to graph the data and results of the computations
for the Task 1 from the coding homeworks in the Machine Learning
course on coursera.com.
"""
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt

import algorithms


def plot_data(data: tuple, title: str = "Scatter Plot Of Training Data") -> None:
    """
    Plot the scatter plot of given x and y arrays. The X axis represents the population of a city in 10,000s, 
    and the Y axis represents the profit in $10,000s. The points are colored based on the label.

    Args:
      data:
        A tuple consisting of x and y arrays.
      title:
        A string that serves as both the plot's title and the saved figure's filename.
    
    Returns:
      None
    """
    x, y = data
    plt.figure(figsize=(10, 6))
    plt.scatter(x[:, 1], y[:, 0], marker="o", c="g", s=100)
    plt.ylim(-5, 25)
    plt.xlim(4, 24)
    plt.ylabel("Profit in $10,000s")
    plt.xlabel("Population of City in 10,000s")
    plt.title(title)
    plt.savefig(title.lower().replace(" ", "_"))


def plot_linear_regression_fit(
    data: tuple, title: str = "Linear Regression Fit"
) -> None:
    """
    Plot the linear regression fit. Linear regression fit is displayed as red line. 
    The points are scattered in the background.
    
    Args:
      data:
        A tuple consisting of x and y arrays.
      title:
        A string that serves as both the plot's title and the saved figure's filename.
    
    Returns:
      None
    """

    def fit(_theta: List, alpha: np.ndarray) -> np.ndarray:
        """
        Compute the linear regression fit. The returned array is the same shape as the 
        input array. Fit has the form: y = mx + b.

        Args:
          _theta:
            A list of theta values.
          alpha:
            An array of the same shape as the input array.
        
        Returns:
          An array of the same shape as the input array.
        """
        return _theta[1] * alpha + _theta[0]

    theta, x, y = data

    plt.figure(figsize=(10, 6))
    plt.plot(x[:, 1], y[:, 0], "rx", markersize=10, label="Training data")
    plt.plot(x[:, 1], fit(theta, x[:, 1]), "b-", label="Linear regression")
    plt.ylabel("Profit in $10,000s")
    plt.xlabel("Population of City in 10,000s")
    plt.ylim(-5, 25)
    plt.xlim(4, 24)
    plt.legend(loc="lower right")
    plt.title(title)
    plt.savefig(title.lower().replace(" ", "_"))


def plot_cost_function_3d(
    data: Tuple[List, List, np.ndarray, np.ndarray],
    title: str = "Surface Plot Of Cost Function",
) -> None:
    """
    Plot the surface plot of the cost function. The surface plot is represented as a 3D surface. 

    Args:
      data:
        A tuple consisting of theta0, theta1, x and y arrays.
      title:
        A string that serves as both the plot's title and the saved figure's filename.
    
    Returns:
      None
    """
    theta_history, _costs, x, y = data

    x_range = np.arange(-10, 10, 0.5)
    y_range = np.arange(-1, 4, 0.2)

    x_values = [x for x in x_range for _ in y_range]
    y_values = [y for _ in x_range for y in y_range]
    z_values = [
        algorithms.compute_cost(x, y, np.array([[x], [y]]))
        for x in x_range
        for y in y_range
    ]

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection="3d")

    ax.scatter(
        x_values, y_values, z_values, c=np.abs(z_values), cmap=plt.get_cmap("YlOrRd")
    )

    plt.plot(
        [theta[0] for theta in theta_history],
        [theta[1] for theta in theta_history],
        _costs,
        "-",
    )

    plt.xlabel(r"$\theta_0$")
    plt.ylabel(r"$\theta_1$")
    plt.title(title)
    plt.savefig(title.lower().replace(" ", "_"))


def plot_convergence(
    data: list, y_lim: Tuple[float, float], title: str = "Convergence Of Cost Function"
) -> None:
    """
    Plot the convergence of the cost function.  The cost function is plotted as a line.

    Args:
      data:
        A list of cost function values.
      y_lim:
        A tuple of the minimum and maximum values of the y axis.
      title:
        A string that serves as both the plot's title and the saved figure's filename.
    
    Returns:
      None
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(data)), data, c="green", s=1)
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost J")
    plt.ylim(y_lim)
    plt.title(title)
    plt.savefig(title.lower().replace(" ", "_"))


def plot_histogram(x: np.ndarray, title: str = "Histogram Of Column Values") -> None:
    """
    Plot the histogram of the column values.
    
    Args:
      x:
        An array of column values.
      title:
        A string that serves as both the plot's title and the saved figure's filename.
    
    Returns:
      None
    """
    plt.figure(figsize=(10, 6))

    for i in range(x.shape[-1]):
        plt.hist(x[:, i], label=f"col{i}")

    plt.xlabel("Column Value")
    plt.ylabel("Counts")
    plt.legend()
    plt.title(title)
    plt.savefig(title.lower().replace(" ", "_"))
