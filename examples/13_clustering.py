import numpy as np
from sklearn.cluster import KMeans
X = np.array([[0.0], [2.0], [8.0], [10.0]])
initial = np.array([[0.0], [10.0]])

def assign(X, centers):
    distances = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return distances.argmin(axis=1)

def update(X, labels, previous):
    centers = previous.copy()
    for k in range(len(centers)):
        members = X[labels == k]
        if len(members):
            centers[k] = members.mean(axis=0)
    return centers

def inertia(X, centers):
    labels = assign(X, centers)
    return np.sum((X - centers[labels]) ** 2)
labels = assign(X, initial)
centers = update(X, labels, initial)
np.testing.assert_array_equal(labels, [0, 0, 1, 1])
np.testing.assert_allclose(centers, [[1.0], [9.0]])
np.testing.assert_allclose([inertia(X, initial), inertia(X, centers)], [8.0, 4.0])
library = KMeans(n_clusters=2, init=initial, n_init=1, algorithm='lloyd', tol=1e-12).fit(X)
np.testing.assert_allclose(np.sort(library.cluster_centers_[:, 0]), [1.0, 9.0])
np.testing.assert_allclose(library.inertia_, inertia(X, centers))
empty = np.array([[0.0], [10.0], [100.0]])
assert np.isfinite(update(X, assign(X, empty), empty)).all()
print('assignments:', labels, 'updated centers:', centers.ravel())
print('inertia/course distortion:', library.inertia_, library.inertia_ / len(X))
