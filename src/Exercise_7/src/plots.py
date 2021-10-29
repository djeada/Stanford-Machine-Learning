""""
The goal of this module is to implement all the visualization
tools needed to graph the data and results of the computations
for the Task 7 from the coding homeworks in the Machine Learning
course on coursera.com.
"""

from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


def plot_data(
    data: Tuple[np.ndarray, np.ndarray], title: str = "Scatter Plot Of Training Data"
) -> None:
    """
    Plots the data in a scatter plot.
    Args:
      data:
        A tuple of x and y values for the points to be plotted.

      title:
        A string that is used as the plot's title and the filename of the saved figure.

    Returns:
      None
    """
    x, y = data
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, marker="o", c="g", s=100)
    plt.title(title)
    plt.savefig(title.lower().replace(" ", "_"))


def visualize_k_means(data: tuple, title="K means algorithm visualisation") -> None:
    """
    Visualizes the k-means algorithm. The algorithm is visualized by plotting the data points 
    in a scatter plot and the centroids in a different color.

    Args:
      data:
        A tuple of x and y values for the points to be plotted.
      title:
        A string that serves as both the plot's title and the saved figure's filename.
    
    Returns:
      None
    """
    x, k, idx, centroid_history = data
    plt.figure(figsize=(10, 6))
    for centroid in range(k):
        row = list()
        for j in range(len(x)):
            if idx[j] == centroid:
                row.append(x[j])

        row = np.array(row)
        plt.scatter(row[:, 0], row[:, 1], marker="o", s=100)

        history = centroid_history[:, centroid, :]
        plt.plot(history[:, 0], history[:, 1], ".-", color="black", markersize=20)

        # ADD SAVEFIG

    plt.title(title)


def visualize_pca(data: Tuple[np.ndarray, np.ndarray]) -> None:
    """
    Visualizes the PCA algorithm. The algorithm is visualized by plotting the 
    data points and the pcas results.

    Args:
      data:
        A tuple of normalized data points and the pca results.
      
    Returns:
      None
    """
    x_norm, x_rec = data
    plt.figure(figsize=(10, 6))
    plt.scatter(
        x_norm[:, 0], x_norm[:, 1], s=30, edgecolors="b", label="Original Data Points"
    )
    plt.scatter(
        x_rec[:, 0], x_rec[:, 1], s=30, edgecolors="r", label="PCA Reduced Data Points"
    )

    for x in range(x_norm.shape[0]):
        plt.plot([x_norm[x, 0], x_rec[x, 0]], [x_norm[x, 1], x_rec[x, 1]], "k--")

    plt.xlabel("x_1 (Feature Normalized)")
    plt.ylabel("x_2 (Feature Normalized)")

    plt.legend()
    plt.title("Original Data Points and Reduced Dimension Points")


def display_image_grid(x: np.ndarray) -> None:
    """
    Displays a grid of images. The images are scaled to the range [0, 1].

    Args:
      x:
        A numpy array of shape (n_images, height, width, channels) containing the images.

    Returns:
      None
    """
    rows, cols = 4, 4
    width, height = 32, 32
    size = int(np.sqrt(x.shape[-1]))
    num_samples = rows * cols
    samples = x[:num_samples]
    image = Image.new("RGB", (rows * width, rows * height))

    for i in range(rows):
        for j in range(cols):
            array = samples[i * rows + j]
            array = ((array / max(array)) * 255).reshape((size, size)).T
            image.paste(Image.fromarray(array + 128), (i * width, j * height))

    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis("off")
