""""
The goal of this module is to implement all the visualization
tools needed to graph the data and results of the computations
for the Task 7 from the coding homeworks in the Machine Learning
course on coursera.com.
"""

from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import algorithms
from PIL import Image


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
    plt.scatter(x, y, marker="o", c="g", s=100)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.show()


def visualize_k_means(x: np.ndarray, model: algorithms.KMeansRegression) -> None:
    """
    Visualizes the KMeans algorithm. The algorithm is visualized by plotting the
    data points and the centroids.

    :param x: The data points.
    :param model: The KMeans model.
    """
    plt.figure(figsize=(10, 6))

    centorid_history = np.array(model.centroids_history)
    clustered_points = {
        tuple(cluster_center): [] for cluster_center in centorid_history[-1]
    }
    for point in x:
        cluster_center = model.predict(point)
        cluster_center = tuple(cluster_center[0])
        clustered_points[cluster_center].append(point)

    for cluster_center, points in clustered_points.items():
        points = np.array(points)
        plt.scatter(
            points[:, 0],
            points[:, 1],
            marker="o",
            s=20,
            label=f"Cluster center: {cluster_center[0]:.2f}, {cluster_center[1]:.2f}",
        )

    for i in range(centorid_history.shape[1]):
        plt.plot(
            centorid_history[:, i, 0],
            centorid_history[:, i, 1],
            marker="x",
            color="black",
        )

    plt.xlabel("x_1")
    plt.ylabel("x_2")
    plt.title("KMeans visualization")
    plt.legend()
    plt.show()


def visualize_pca(
    x: np.ndarray,
    x_recovered: np.ndarray,
    title: str = "PCA visualization",
    x_label: str = "x_1 (Feature Normalized)",
    y_label: str = "x_2 (Feature Normalized)",
) -> None:
    """
    Visualizes the PCA algorithm. The algorithm is visualized by plotting the
    data points and the pcas results.

    :param x: The original data points.
    :param x_recovered: The data points after PCA.
    """
    plt.figure(figsize=(10, 6))

    # scatter plot the original data points and the recovered data points
    plt.scatter(x[:, 0], x[:, 1], marker="o", c="g", s=20, label="Original Data")
    plt.scatter(
        x_recovered[:, 0], x_recovered[:, 1], marker="o", c="b", s=20, label="PCA"
    )

    # draw the line between the original and the recovered data points
    for i in range(x.shape[0]):
        plt.plot([x[i, 0], x_recovered[i, 0]], [x[i, 1], x_recovered[i, 1]], "k--")

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()


def display_image_grid(
    x: np.ndarray,
    rows: int = 4,
    cols: int = 4,
    width: int = 32,
    height: int = 32,
    title: str = "Image Grid",
) -> None:
    """
    Displays a grid of images.

    :param x: The images to be displayed.
    :param rows: The number of rows in the grid.
    :param cols: The number of columns in the grid.
    :param width: The width of the grid.
    :param height: The height of the grid.
    :param title: The title of the grid.
    """

    plt.figure(figsize=(6, 6))
    plt.gray()
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(x[i].reshape(width, height).T)
        plt.axis("off")
    plt.suptitle(title)
    plt.show()


def plot_image(image: Image, title: str = "Image") -> None:
    """
    Plots a PIL image.

    :param image: The image to be plotted.
    :param title: The title of the plot.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.title(title)
    plt.show()
