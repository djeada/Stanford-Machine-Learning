""""
The goal of this module is to implement all algorithms and numerical
methods needed to solve the Task 7 from the coding homeworks in the
Machine Learning course on coursera.com.
"""
from typing import Optional, Tuple

import numpy as np
from PIL import Image


class KMeansRegression:
    """
    The K-Means Regression algorithm.

    :param k: The number of clusters.
    """

    def __init__(self, k: int) -> None:
        self.k = k
        self.centroids_history = []

    def fit(
        self,
        x: np.ndarray,
        max_iterations: int = 10,
        initial_centroids: Optional[np.ndarray] = None,
    ) -> None:
        """
        Fits the model to the data.

        :param x: The data points.
        :param max_iterations: The maximum number of iterations.
        :param initial_centroids: The initial centroids.
        """
        if initial_centroids is not None:
            centroids = initial_centroids
        else:
            centroids = self._initialize_centroids(x)

        self.centroids_history = [centroids]
        for _ in range(max_iterations):
            idx = self._find_closest_centroids(x, centroids)
            centroids = self._compute_centroids(x, idx)
            self.centroids_history.append(centroids)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Accepts an array of data points and returns an array of cluster centroids
        for each data point.

        :param x: The data points.
        :return: The cluster centroids.
        """
        x = np.atleast_2d(x)
        return self.centroids_history[-1][
            self._find_closest_centroids(x, self.centroids_history[-1])
        ]

    def _initialize_centroids(self, x: np.ndarray) -> np.ndarray:
        """
        Initializes the centroids.

        :param x: The data points.
        :return: The centroids.
        """
        return x[np.random.choice(x.shape[0], self.k, replace=False)]

    def _find_closest_centroids(
        self, x: np.ndarray, centroids: np.ndarray
    ) -> np.ndarray:
        """
        Finds the closest centroid for each data point.

        :param x: The data points.
        :param centroids: The centroids.
        :return: The cluster index of each data point.
        """
        return np.argmin(np.linalg.norm(x[:, np.newaxis] - centroids, axis=2), axis=1)

    def _compute_centroids(self, x: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """
        Computes the centroids.

        :param x: The data points.
        :param idx: The cluster index of each data point.
        :return: The centroids.
        """
        return np.array(
            [x[idx == centroid_index].mean(axis=0) for centroid_index in range(self.k)]
        )


class PcaRegression:
    """
    The PCA Regression algorithm.

    :param k: The number of eigenvectors.
    """

    def __init__(self, k: int) -> None:
        self.k = k
        self.u = None
        self.z = None

    def fit(self, x: np.ndarray) -> None:
        """
        Fits the model to the data.

        :param x: The data points.
        """
        sigma = (1 / x.shape[0]) * x.T @ x
        self.u, _, _ = np.linalg.svd(sigma)
        u_reduce = self.u[:, : self.k]
        self.z = x @ u_reduce

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Accepts an array of data points and returns an array of cluster centroids
        for each data point.

        :param x: The data points.
        :return: The cluster centroids.
        """
        x = np.atleast_2d(x)
        u_reduce = self.u[:, : self.k]
        z = x @ u_reduce
        return z

    def recover_data(self) -> np.ndarray:
        """
        Recovers the original data.

        :return: The original data.
        """
        return self.z @ self.u[:, : self.k].T


def reconstruct_image(
    model: KMeansRegression, x: np.ndarray, original_shape: Tuple[int, int]
) -> Image:
    """
    Reconstructs the image from the compressed representation.

    :param model: The K-Means Regression model.
    :param x: The compressed representation.
    :param original_shape: The original shape of the image.
    :return: The reconstructed image.
    """
    centorid_history = np.array(model.centroids_history)
    clustered_points = {
        tuple(cluster_center): [] for cluster_center in centorid_history[-1]
    }
    for point in x:
        cluster_center = model.predict(point)
        cluster_center = tuple(cluster_center[0])
        clustered_points[cluster_center].append(point)
    centroids = np.array(
        [
            np.mean(clustered_points[cluster_center], axis=0)
            for cluster_center in clustered_points
        ]
    )
    idx = [np.argmin(np.linalg.norm(point - centroids, axis=1)) for point in x]
    x_reconstructed = np.array(centroids[idx, :] * 255, dtype=np.uint8).reshape(
        original_shape
    )
    return Image.fromarray(x_reconstructed)
