""""
The goal of this module is to implement all the visualization
tools needed to graph the data and results of the computations
for the Task 2 from the coding homeworks in the Machine Learning
course on coursera.com.
"""
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import algorithms


def plot_data(
    data: Tuple[np.ndarray, np.ndarray],
    labels: Tuple[str, str, str, str],
    title: str = "scatter plot of training data",
) -> None:
    """
    Scatter plot of training data.

    Args:
      data:
        A tuple of x and y values for the points to be plotted.
      labels:
        A tuple of four strings that are used for axis labels and legend.
      title:
        A string that serves as both the plot's title and the saved figure's filename.

    Returns:
      None
    """
    x, y = data
    label_pos, label_neg, label_x, label_y = labels
    plt.figure(figsize=(10, 6))
    plt.scatter(x[:, 1], x[:, 2], marker="+", c="k", s=150, label=label_pos)
    plt.scatter(y[:, 1], y[:, 2], marker="o", c="y", s=100, label=label_neg)
    plt.ylabel(label_y)
    plt.xlabel(label_x)
    plt.legend(loc="upper right")
    plt.title(title)
    plt.savefig(title.lower().replace(" ", "_"))


def plot_logistic_regression_fit(
    data: tuple, title: str = "logistic regression fit"
) -> None:
    """
    Plot logistic regression fit.

    Args:
      data:
        A tuple consisting of theta values as well as x and y sets.
      title:
        A string that serves as both the plot's title and the saved figure's filename.

    Returns:
      None
    """

    def fit(_theta: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """
        Compute the cost function for the logistic regression. 
        This function is used to plot the cost function as a function of theta.
        The logistic regression fit has the form: y_i = -1/theta_2*(alpha_i*theta_1 + theta_0)

        Args:
          _theta:
            A list of three parameters used in the linear hypothesis function.
          alpha:
            An array of x values that are used in the linear fit.

        Returns:
          An array of Y values.
        """
        return (-1.0 / _theta[2]) * (_theta[0] + _theta[1] * alpha)

    theta, x, y, pos, neg = data
    x_range = np.array([np.min(x[:, 1]), np.max(x[:, 1])])

    plt.figure(figsize=(10, 6))
    plt.scatter(pos[:, 1], pos[:, 2], marker="+", c="k", s=150, label="Admitted")
    plt.scatter(neg[:, 1], neg[:, 2], marker="o", c="y", s=100, label="Not admitted")
    plt.plot(x_range, fit(theta, x_range), "b-")
    plt.ylabel("Exam 2 score")
    plt.xlabel("Exam 1 score")
    plt.legend(loc="upper right")
    plt.title(title)
    plt.savefig(title.lower().replace(" ", "_"))


def plot_boundary(
    data: Tuple[np.ndarray, int], title: str = "Decision Boundary"
) -> None:
    """"
    Plot a decision boundary on top of the dataset. A decision boundary is
    the line that divides two classes of data. Decision boundary occurs
    when the hypothesis function is equal to zero.

    Args:
      data:
        A tuple consisting  of an array of theta values and a single lambda value.
      title:
        A string that serves as both the plot's title and the saved figure's filename.

    Returns:
      None
    """

    theta, _lambda = data

    range_x = np.linspace(-1, 1.5)
    range_y = np.linspace(-1, 1.5)
    range_z = [
        [
            np.dot(theta, algorithms.map_feature(np.array([y]), np.array([x])).T)[0][0]
            for x in range_x
        ]
        for y in range_y
    ]
    range_z = np.array(range_z).T

    contour = plt.contour(
        range_x, range_y, range_z, [0], cmap=plt.cm.coolwarm, extend="both"
    )
    plt.clabel(contour, inline=1, fmt={0: f"Lambda = {_lambda}"})
    plt.title(title)
    plt.savefig(title.lower().replace(" ", "_"))
