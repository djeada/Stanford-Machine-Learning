""""
The goal of this module is to implement all the visualization
tools needed to graph the data and results of the computations
for the Task 1 from the coding homeworks in the Machine Learning
course on coursera.com.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import algorithms


def plot_data(data: tuple) -> None:
    """
    """
    x, y = data
    plt.figure(figsize=(10, 6))
    plt.scatter(x[:, 1], y[:, 0], marker="o", c="g", s=100)
    plt.ylim(-5, 25)
    plt.xlim(4, 24)
    plt.ylabel("Profit in $10,000s")
    plt.xlabel("Population of City in 10,000s")
    plt.savefig("scatter_plot_of_training_data")


def plot_linear_regression_fit(data) -> None:
    def fit(theta, alpha):
        return theta[0] + theta[1] * alpha

    theta, x, y = data

    plt.figure(figsize=(10, 6))
    plt.plot(x[:, 1], y[:, 0], "rx", markersize=10, label="Training data")
    plt.plot(x[:, 1], fit(theta, x[:, 1]), "b-", label="Linear regression")
    plt.ylabel("Profit in $10,000s")
    plt.xlabel("Population of City in 10,000s")
    plt.ylim(-5, 25)
    plt.xlim(4, 24)
    plt.legend(loc="lower right")
    plt.savefig("linear_regression_fit")


def plot_cost_function_3d(data: tuple) -> None:
    """
    """
    theta_history, _costs, x, y = data

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection="3d")

    x_range = np.arange(-10, 10, 0.5)
    y_range = np.arange(-1, 4, 0.2)

    x_values = [x for x in x_range for y in y_range]
    y_values = [y for x in x_range for y in y_range]
    z_values = [
        algorithms.compute_cost(x, y, np.array([[x], [y]])) for x in x_range for y in y_range
    ]

    ax.scatter(
        x_values, y_values, z_values, c=np.abs(z_values), cmap=plt.get_cmap("YlOrRd")
    )

    plt.plot(
        [theta[0] for theta in theta_history], [theta[1] for theta in theta_history], _costs, "-"
    )

    plt.xlabel(r"$\theta_0$")
    plt.ylabel(r"$\theta_1$")
    plt.title("Cost Function")
    plt.savefig("surface_plot_of_cost_function")


def plot_convergence(data: list, y_lim: tuple, output_file: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(data)), data, c="green", s=1)
    plt.title("Convergence of Cost Function")
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost J")
    plt.ylim(y_lim)
    plt.savefig(output_file)


def plot_histogram(x: np.ndarray, output_file: Path) -> None:
    """
    """
    plt.figure(figsize=(10, 6))

    for i in range(x.shape[-1]):
        plt.hist(x[:, i], label=f"col{i}")

    plt.xlabel("Column Value")
    plt.ylabel("Counts")
    plt.legend()
    plt.savefig(output_file)
