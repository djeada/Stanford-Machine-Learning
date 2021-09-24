""""
The goal of this module is to implement all algorithms and numerical
methods needed to solve the Task 1 from the coding homeworks in the
Machine Learning course on coursera.com.
"""

import numpy as np

def hypothesis_function(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return np.dot(x, theta)


def compute_cost(x: np.ndarray, y: np.ndarray, theta: np.ndarray = None) -> np.float64:
    if theta is None:
        theta = np.zeros((x.shape[1], 1))

    return (
            1.0
            / (2 * y.size)
            * np.dot(
        (hypothesis_function(x, theta) - y).T, (hypothesis_function(x, theta) - y)
    )
    )[0][0]


def gradient_descent(x: np.ndarray, y: np.ndarray, theta: np.ndarray = None, num_iter: int = 2000,
                     alpha: float = 0.01) -> tuple:
    if theta is None:
        theta = np.zeros((x.shape[1], 1))

    initial_theta = theta
    m = y.size
    costs = []
    theta_history = []

    for _ in range(num_iter):
        costs.append(compute_cost(x, y, theta))
        theta_history.append(list(theta[:, 0]))

        for i in range(len(theta)):
            theta[i] -= (alpha / m) * np.sum(
                (hypothesis_function(x, initial_theta) - y)
                * np.array(x[:, i]).reshape(m, 1)
            )

    return theta_history, costs

def normalize_features(x: np.ndarray) -> tuple:
    means = [np.mean(x[:, 0])]
    stds = [np.std(x[:, 0])]

    for i in range(1, x.shape[1]):
        means.append(np.mean(x[:, i]))
        stds.append(np.std(x[:, i]))
        x[:, i] = (x[:, i] - means[-1]) / stds[-1]

    return means, stds

def predict_from_means_and_stds(means: list, stds: list, theta: np.ndarray, house_size: int,
                                num_bedrooms: int) -> np.ndarray:
    y = [house_size, num_bedrooms]
    y = [(y[i] - means[i + 1]) / stds[i + 1] for i in range(len(y))]
    y.insert(0, 1)
    y = np.array(y)

    return hypothesis_function(y, theta)

def predict_from_normal_equation(x: np.ndarray, y: np.ndarray, house_size: int, num_bedrooms: int) -> np.ndarray:
    def norm_eq(_x, _y):
        return np.dot(np.dot(np.linalg.inv(np.dot(_x.T, _x)), _x.T), _y)

    return hypothesis_function(np.array([1, house_size, num_bedrooms]), norm_eq(x, y))