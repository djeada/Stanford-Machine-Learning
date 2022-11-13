""""
The goal of this module is to implement all the visualization
tools needed to graph the data and results of the computations
for the Task 8 from the coding homeworks in the Machine Learning
course on coursera.com.
"""

import numpy as np
import matplotlib.pyplot as plt

import algorithms


def plot_data(data: tuple, title: str = "scatter plot of training data") -> None:
    x, y = data
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, marker="x", c="g", s=100)
    plt.xlabel("Latency (ms)")
    plt.ylabel("Throughput (mb/s)")
    plt.title(title)
    plt.savefig(title.lower().replace(" ", "_"))


def plot_gaussian_contours(data: tuple, n: int = 200) -> None:
    x, x_min, x_max, y_min, y_max = data

    x_range = np.array(np.linspace(x_min, x_max, n))
    y_range = np.array(np.linspace(y_min, y_max, n))
    u, v = np.meshgrid(x_range, y_range)
    grid = np.array(list(zip(u.flatten(), v.flatten())))

    mu, sigma_2 = algorithms.get_gaussian_parameters(x)
    z = algorithms.compute_gauss(grid, mu, sigma_2)
    z = z.reshape(u.shape)

    plt.contour(
        x_range,
        y_range,
        z,
        np.array([10.0]) ** np.arange(-21, 0, 3.0),
        cmap=plt.cm.coolwarm,
        extend="both",
    )


def plot_anomalies(x: np.ndarray, best_eps: float) -> None:
    gauss_values = algorithms.compute_gauss(x, *algorithms.get_gaussian_parameters(x))
    anomalies = np.array(
        [x[i] for i in range(x.shape[0]) if gauss_values[i] < best_eps]
    )
    plt.scatter(anomalies[:, 0], anomalies[:, 1], s=200, color="red", marker="x")


def plot_movies_data(y: np.ndarray, title: str = "movies data") -> None:
    plt.figure(figsize=(10, 6))
    plt.imshow(y, aspect="auto")
    plt.colorbar()
    plt.ylabel(f"Movies {y.shape[0]}", fontsize=20)
    plt.xlabel(f"Users {y.shape[-1]}", fontsize=20)
    plt.title(title)
    plt.savefig(title.lower().replace(" ", "_"))
