""""
The goal of this module is to implement all the visualization
tools needed to graph the data and results of the computations
for the Task 8 from the coding homeworks in the Machine Learning
course on coursera.com.
"""

import numpy as np
import matplotlib.pyplot as plt

import algorithms


def scatter(
    x: np.ndarray,
    y: np.ndarray,
    title: str = "Training Data",
    x_label: str = "x",
    y_label: str = "y",
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
    plt.scatter(x, y, marker="x", c="black", s=100)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.show()


def plot_gaussian_contours(
    model: algorithms.GaussianRegression,
    x: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    n: int = 100,
) -> None:
    """
    Plots the contours of the Gaussian distribution.

    :param model: The GaussianRegression model.
    :param x: The data points.
    :param x_min: The minimum value of the x axis.
    :param x_max: The maximum value of the x axis.
    :param y_min: The minimum value of the y axis.
    :param y_max: The maximum value of the y axis.
    :param n: The number of points to use for the grid.
    """
    plt.figure(figsize=(10, 6))

    # plot the data
    plt.scatter(x[:, 0], x[:, 1], marker="x", c="black", s=100)

    # plot the contours
    x_range = np.array(np.linspace(x_min, x_max, n))
    y_range = np.array(np.linspace(y_min, y_max, n))
    u, v = np.meshgrid(x_range, y_range)
    grid = np.array(list(zip(u.flatten(), v.flatten())))

    z = model.predict(grid)
    z = z.reshape(u.shape)

    plt.contour(
        x_range,
        y_range,
        z,
        np.array([10.0]) ** np.arange(-21, 0, 3.0),
        cmap=plt.cm.coolwarm,
        extend="both",
    )
    plt.show()


def plot_anomalies(
    model: algorithms.GaussianRegression,
    x: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    n: int = 100,
) -> None:
    """
    Plots the contours of the Gaussian distribution with the anomalies.

    :param model: The GaussianRegression model.
    :param x: The data points.
    :param x_min: The minimum value of the x axis.
    :param x_max: The maximum value of the x axis.
    :param y_min: The minimum value of the y axis.
    :param y_max: The maximum value of the y axis.
    :param n: The number of points to use for the grid.
    """
    plt.figure(figsize=(10, 6))

    # plot the data
    plt.scatter(x[:, 0], x[:, 1], marker="x", c="black", s=100)

    # use model.is_anomaly to find the anomalies
    anomalies = x[model.is_anomaly(x)]
    plt.scatter(anomalies[:, 0], anomalies[:, 1], s=200, color="red", marker="x")

    # plot the contours
    x_range = np.array(np.linspace(x_min, x_max, n))
    y_range = np.array(np.linspace(y_min, y_max, n))
    u, v = np.meshgrid(x_range, y_range)
    grid = np.array(list(zip(u.flatten(), v.flatten())))

    z = model.predict(grid)
    z = z.reshape(u.shape)

    plt.contour(
        x_range,
        y_range,
        z,
        np.array([10.0]) ** np.arange(-21, 0, 3.0),
        cmap=plt.cm.coolwarm,
        extend="both",
    )

    plt.show()


def plot_movie_data(y: np.ndarray, title: str = "Movie data") -> None:
    """
    Plots the movie data.

    :param y: The movie data.
    :param title: The title of the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(y, aspect="auto")
    plt.colorbar()
    plt.ylabel(f"Movies {y.shape[0]}", fontsize=20)
    plt.xlabel(f"Users {y.shape[-1]}", fontsize=20)
    plt.title(title)
    plt.savefig(title.lower().replace(" ", "_"))
