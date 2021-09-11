import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from PIL import Image

DATA_PATH_1 = "data/data2.mat"
DATA_PATH_2 = "data/bird_small.png"
DATA_PATH_3 = "data/data1.mat"
DATA_PATH_4 = "data/faces.mat"


def read_data(path):
    """
    x is a matrix with m rows and n columns
    """

    data = sio.loadmat(path)
    x = data["X"]

    return x


def read_image(path):
    x = np.asarray(Image.open(path)) / 255
    original_shape = x.shape
    x = x.reshape(-1, 3)

    return x, original_shape


def initialize_k_centroids(x, k):
    """
    pick random k rows from x
    """
    n = x.shape[0]
    return x[np.random.choice(n, k, replace=False), :]


def find_closest_centroids(x, centroids):
    """
    implementation of cluster assignment step in K-means algorithm
    """
    closest_cluster = [np.argmin(np.linalg.norm(row - centroids, axis=1)) for row in x]
    return closest_cluster


def compute_means(x, idx, k):
    """
    implementation of move centroid step in K-means algorithm
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


def reconsruct_image(centroids, idx, original_shape):
    idx = np.array(idx, dtype=np.uint8)
    x_reconstructed = np.array(centroids[idx, :] * 255, dtype=np.uint8).reshape(
        original_shape
    )
    return Image.fromarray(x_reconstructed)


def find_k_means(x, k, max_iters=10):

    idx = None
    centroid_history = list()
    centroids = initialize_k_centroids(x, k)

    for i in range(max_iters):
        idx = find_closest_centroids(x, centroids)
        centroids = compute_means(x, idx, k)
        centroid_history.append(centroids)

    return np.array(centroid_history), idx


def normalize_features(x):
    feature_means = np.mean(x, axis=0)
    feature_stds = np.std(x, axis=0)
    x_norm = (x - feature_means) / feature_stds
    return x_norm, feature_means, feature_stds


def pca(x):
    m = len(x)
    sigma = (1 / m) * x.T @ x
    u, s, _ = np.linalg.svd(sigma)
    return u, s


def project_data(x, u, k):
    u_reduce = u[:, :k]
    z = x @ u_reduce
    return z


def recover_data(z, u, k):
    u_reduce = u[:, :k]
    x_rec = u_reduce @ z.T
    return x_rec.T


def plot_data(data, title="scatter plot of training data"):
    x, y = data
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, marker="o", c="g", s=100)
    plt.title(title)
    plt.savefig(title.lower().replace(" ", "_"))


def visualize_k_means(data, title="K means algorithm visualisation"):
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

    plt.title(title)


def visualize_pca(data):
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


def display_image_grid(x):
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


def part_1():
    x = read_data(DATA_PATH_1)
    plot_data((x[:, 0], x[:, 1]))

    k = 3
    initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
    idx = find_closest_centroids(x, initial_centroids)
    centroids = compute_means(x, idx, k)
    print(centroids)
    centroid_history, idx = find_k_means(x, k)
    visualize_k_means((x, k, idx, centroid_history))


def part_2():
    x, original_shape = read_image(DATA_PATH_2)
    k = 16
    centroid_history, _ = find_k_means(x, k, max_iters=10)
    colors = centroid_history[-1]
    print(colors.shape)
    idx = find_closest_centroids(x, colors)

    compressed_image = reconsruct_image(colors, idx, original_shape)
    compressed_image.show()


def part_3():
    x = read_data(DATA_PATH_3)
    plot_data((x[:, 0], x[:, 1]), title="scatter plot of training data 2")
    x_norm, _, _ = normalize_features(x)

    u, _ = pca(x_norm)
    print(u)

    k = 1
    z = project_data(x_norm, u, k)
    print(z[0])
    x_rec = recover_data(z, u, k)

    visualize_pca((x_norm, x_rec))


def part_4():
    x = read_data(DATA_PATH_4)
    plot_data((x[:, 0], x[:, 1]))
    display_image_grid(x)

    k = 100
    x_norm, _, _ = normalize_features(x)
    u, _ = pca(x_norm)
    z = project_data(x_norm, u, k)
    x_rec = recover_data(z, u, k)

    display_image_grid(x_rec)


def main():
    plt.style.use("seaborn")
    part_1()
    part_2()
    part_3()
    part_4()
    plt.show()


if __name__ == "__main__":
    main()
