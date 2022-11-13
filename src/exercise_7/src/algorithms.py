""""
The goal of this module is to implement all algorithms and numerical
methods needed to solve the Task 7 from the coding homeworks in the
Machine Learning course on coursera.com.
"""
from typing import Optional, Tuple

import numpy as np
from PIL import Image


def initialize_k_centroids(x: np.ndarray, k: int) -> np.ndarray:
    """
    Pick k random points as initial centroids. This is the first step in K-means algorithm.

    Args:
        x: the dataset
        k: the number of centroids

    Returns:
        the initial centroids
    """
    n = x.shape[0]
    return x[np.random.choice(n, k, replace=False), :]


def find_closest_centroids(x: np.ndarray, centroids: np.ndarray) -> list:
    """
    Implementation of cluster assignment step in K-means algorithm. 
    This function returns the list of centroid assignments for each data point.

    Args:
        x: the dataset
        centroids: the centroids
    
    Returns:
        the list of centroid assignments for each data point.
    """
    closest_cluster = [np.argmin(np.linalg.norm(row - centroids, axis=1)) for row in x]
    return closest_cluster


def compute_means(x: np.ndarray, idx: list, k: int) -> list:
    """
    Implementation of the mean update step in K-means algorithm.

    Args:
        x: the dataset
        idx: the list of centroid assignments for each data point
        k: the number of centroids

    Returns:
        the updated centroids
    """
    rows = list()

    for i in range(k):
        row = list()
        for j in range(len(x)):
            if idx[j] == i:
                row.append(x[j])
        rows.append(np.array(row))

    centroids = [[np.mean(column) for column in row.T] for row in rows]
    return centroids


def reconstruct_image(
    centroids: np.ndarray, idx: list, original_shape: tuple[int]
) -> Image:
    """
    Reconstruct an image from its compressed representation. The compressed representation is given by the indices of the centroids.

    Args:
        centroids: the centroids
        idx: the list of centroid assignments for each data point
        original_shape: the original shape of the image
    
    Returns:
        the reconstructed image
    """
    idx = np.array(idx, dtype=np.uint8)
    x_reconstructed = np.array(centroids[idx, :] * 255, dtype=np.uint8).reshape(
        original_shape
    )
    return Image.fromarray(x_reconstructed)


def find_k_means(
    x: np.ndarray, k: int, max_iter: int = 10
) -> Tuple[np.ndarray, Optional[list]]:
    """
    Implementation of K-means algorithm. This function returns the centroids and the list of centroid assignments for each data point.

    Args:
        x: the dataset
        k: the number of centroids
        max_iter: the maximum number of iterations
    
    Returns:
        the centroids and the list of centroid assignments for each data point.
    """
    idx = None
    centroid_history = list()
    centroids = initialize_k_centroids(x, k)

    for i in range(max_iter):
        idx = find_closest_centroids(x, centroids)
        centroids = compute_means(x, idx, k)
        centroid_history.append(centroids)

    return np.array(centroid_history), idx


def normalize_features(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize the features in x. This function returns the normalized features and the mean used for normalization.

    Args:
        x: the dataset
    
    Returns:
        the normalized features and the mean used for normalization.
    """
    feature_means = np.mean(x, axis=0)
    feature_stds = np.std(x, axis=0)
    x_norm = (x - feature_means) / feature_stds
    return x_norm, feature_means, feature_stds


def pca(x: np.ndarray) -> tuple:
    """
    Implementation of PCA. This function returns the eigenvectors and eigenvalues of the covariance matrix.

    Args:
        x: the dataset
    
    Returns:
        the eigenvectors and eigenvalues of the covariance matrix.
    """
    m = len(x)
    sigma = (1 / m) * x.T @ x
    u, s, _ = np.linalg.svd(sigma)
    return u, s


def project_data(x: np.ndarray, u: np.ndarray, k: int) -> np.ndarray:
    """
    Project the data onto the first k eigenvectors.

    Args:
        x: the dataset
        u: the eigenvectors
        k: the number of eigenvectors
    
    Returns:
        the projected data
    """
    u_reduce = u[:, :k]
    z = x @ u_reduce
    return z


def recover_data(z: np.ndarray, u: np.ndarray, k: int) -> np.ndarray:
    """
    Recover the data from the compressed representation given by the first k eigenvectors.

    Args:
        z: the compressed representation
        u: the eigenvectors
        k: the number of eigenvectors
    
    Returns:
        the recovered data
    """
    u_reduce = u[:, :k]
    x_rec = u_reduce @ z.T
    return x_rec.T
