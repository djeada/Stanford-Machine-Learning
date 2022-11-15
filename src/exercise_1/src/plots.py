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


def scatter(
    x: np.ndarray, y: np.ndarray, title: str, x_label: str = "x", y_label: str = "y"
) -> None:
    """
    Plots the data as a scatter plot.

    :param x: The x values.
    :param y: The y values.
    :param title: The title of the plot.
    :param x_label: The label of the x axis.
    :param y_label: The label of the y axis.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, marker="x", c="r", s=100)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.show()


def plot_regression_fit(
    x: np.ndarray,
    y: np.ndarray,
    model: algorithms.LinearRegressionBase,
    title: str = "Training Data With Linear Regression Fit",
    x_label="x",
    y_label="y",
) -> None:

    """
    Plots the training data and the regression fit.

    :param x: The x values of the data.
    :param y: The y values of the data.
    :param model: The model used to fit the data.
    :param title: The title of the plot.
    :param x_label: The label of the x axis.
    :param y_label: The label of the y axis.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(x[:, 1], y, marker="x", c="r", s=100)
    plt.plot(x[:, 1], model.predict(x), c="b")
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    # add legend with string representation of the model
    plt.legend(["Training Data", str(model)])
    plt.show()


def plot_cost_function(
    cost_history: np.ndarray, title: str = "Cost Function Convergence"
) -> None:
    """
    Plot the cost function convergence.

    :param cost_history: The cost function history.
    :param title: The title of the plot.
    """

    plt.figure(figsize=(10, 6))
    plt.scatter(np.arange(cost_history.size), cost_history, marker="o", c="g", s=10)
    plt.xlabel("Iterations")
    plt.ylabel("Cost J")
    plt.title(title)
    plt.show()


def plot_cost_function_3d(
    model: algorithms.LinearRegressionBase,
    x: np.ndarray,
    y: np.ndarray,
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
    model = model.copy()  # copy the model to avoid changing the original model
    theta_history, cost_history = np.array(model.theta_history), model.cost_history

    x_range = np.arange(-1, 4, 0.2)
    y_range = np.arange(-10, 10, 0.5)

    x_values = [x for x in x_range for _ in y_range]
    y_values = [y for _ in x_range for y in y_range]
    z_values = []
    for x_val, y_val in zip(x_values, y_values):
        model.theta = np.array([y_val, x_val])
        z_values.append(model.cost(x, y))

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection="3d")
    ax.invert_xaxis()

    # use gradient for plot color
    ax.plot_surface(
        np.array(x_values).reshape(len(x_range), len(y_range)),
        np.array(y_values).reshape(len(x_range), len(y_range)),
        np.array(z_values).reshape(len(x_range), len(y_range)),
        cmap="viridis",
        alpha=0.5,
    )

    plt.scatter(
        theta_history[:, 1],
        theta_history[:, 0],
        cost_history,
        c="r",
        marker="x",
        label="Gradient Descent",
    )

    plt.xlabel(r"$\theta_0$")
    plt.ylabel(r"$\theta_1$")
    ax.set_zlabel("Cost J")
    plt.legend()
    plt.title(title)
    plt.show()


def plot_histogram(x: np.ndarray, title: str = "Histogram") -> None:
    """
    Plots the histogram of the data.

    :param x: The data.
    :param title: The title of the plot.
    """
    plt.figure(figsize=(10, 6))

    for i in range(x.shape[-1]):
        plt.hist(x[:, i], label=f"col{i}")

    plt.xlabel("Column Value")
    plt.ylabel("Counts")
    plt.legend()
    plt.title(title)
    plt.show()
