""""
The goal of this module is to give a comprehensive solution to Task 7
from the coding homeworks from the Machine Learning course on coursera.com.
The task is broken down into smaller parts.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import readers
import algorithms
import plots

DATA_PATH_1 = Path("../data/data2.mat")
DATA_PATH_2 = Path("../data/bird_small.png")
DATA_PATH_3 = Path("../data/data1.mat")
DATA_PATH_4 = Path("../data/faces.mat")


def part_1():
    """
    K-means Clustering:
    - Implementing K-means.
    - Finding closest centroids.
    - Computing centroid means.

    Returns:
      None
    """
    x = readers.read_data(DATA_PATH_1)
    plots.plot_data((x[:, 0], x[:, 1]))

    k = 3
    initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
    idx = algorithms.find_closest_centroids(x, initial_centroids)
    centroids = algorithms.compute_means(x, idx, k)
    print(centroids)
    centroid_history, idx = algorithms.find_k_means(x, k)
    plots.visualize_k_means((x, k, idx, centroid_history))


def part_2():
    """
    K-means Clustering:
    - Random initialization.
    - Image compression with K-means.

    Returns:
      None
    """
    x, original_shape = readers.read_image(DATA_PATH_2)
    k = 16
    centroid_history, _ = algorithms.find_k_means(x, k)
    colors = centroid_history[-1]
    print(colors.shape)
    idx = algorithms.find_closest_centroids(x, colors)

    compressed_image = algorithms.reconstruct_image(colors, idx, original_shape)
    compressed_image.show()


def part_3():
    """
    Principal Component Analysis:
    - Implementing PCA.
    - Dimensionality Reduction with PCA.
    - Visualizing the projections.

    Returns:
      None
    """
    x = readers.read_data(DATA_PATH_3)
    plots.plot_data((x[:, 0], x[:, 1]), title="scatter plot of training data 2")
    x_norm, _, _ = algorithms.normalize_features(x)

    u, _ = algorithms.pca(x_norm)
    print(u)

    k = 1
    z = algorithms.project_data(x_norm, u, k)
    print(z[0])
    x_rec = algorithms.recover_data(z, u, k)

    plots.visualize_pca((x_norm, x_rec))


def part_4():
    """
    Principal Component Analysis:
    - PCA on faces.
    - Dimensionality reduction.

    Returns:
      None
    """
    x = readers.read_data(DATA_PATH_4)
    plots.plot_data((x[:, 0], x[:, 1]))
    plots.display_image_grid(x)

    k = 100
    x_norm, _, _ = algorithms.normalize_features(x)
    u, _ = algorithms.pca(x_norm)
    z = algorithms.project_data(x_norm, u, k)
    x_rec = algorithms.recover_data(z, u, k)

    plots.display_image_grid(x_rec)


def main() -> None:
    """
    The main functions. Calls the functions that implement different parts of the solution
    to the Task 7 from the coding homeworks from the Machine Learning course on coursera.com.

    Returns:
      None
    """
    plt.style.use("seaborn")
    part_1()
    part_2()
    part_3()
    part_4()
    plt.show()


if __name__ == "__main__":
    main()
