""""
The goal of this module is to implement all algorithms and numerical
methods needed to solve the Task 2 from the coding homeworks in the
Machine Learning course on coursera.com.
"""

import numpy as np
from scipy import optimize


def hypothesis_function(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    def sigmoid(z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    return sigmoid(np.dot(x, theta))


def compute_cost(x: np.ndarray, y: np.ndarray, theta: np.ndarray = None, _lambda: int = 0) -> np.float64:
    if theta is None:
        theta = np.zeros((x.shape[1], 1))

    a = np.dot(-np.array(y).T, np.log(hypothesis_function(x, theta)))
    b = np.dot((1 - np.array(y)).T, np.log(1 - hypothesis_function(x, theta)))
    c = (_lambda / 2) * np.sum(np.dot(theta[1:].T, theta[1:]))

    return (1.0 / y.size) * (np.sum(a - b) + c)


def optimize_theta(x: np.ndarray, y: np.ndarray, theta: np.ndarray = None, _lambda: int = 0) -> np.ndarray:
    if theta is None:
        theta = np.zeros((x.shape[1], 1))

    return optimize.fmin(
        lambda _theta: compute_cost(x, y, _theta, _lambda),
        x0=theta,
        maxiter=400,
        full_output=True,
    )[0]


def predict(theta: np.ndarray, score_1: int, score_2: int) -> np.float64:
    return hypothesis_function(theta, np.array([1, score_1, score_2]))


def correct_predictions(theta: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> float:
    total = 0

    for score in pos:
        if predict(theta, score[1], score[2]) >= 0.5:
            total += 1

    for score in neg:
        if predict(theta, score[1], score[2]) < 0.5:
            total += 1

    return total / (len(pos) + len(neg))


def map_feature(x1_col: np.ndarray, x2_col: np.ndarray) -> np.ndarray:
    degrees = 6
    result = np.ones((x1_col.shape[0], 1))

    for i in range(degrees):
        for j in range(i + 2):
            a = x1_col ** (i - j + 1)
            b = x2_col ** j
            result = np.hstack((result, (a * b).reshape(a.shape[0], 1)))
    return result


def optimize_regularized_theta(x: np.ndarray, y: np.ndarray, theta: np.ndarray = None, _lambda: int = 0) -> np.ndarray:
    if theta is None:
        theta = np.zeros((x.shape[1], 1))

    result = optimize.minimize(
        lambda _theta: compute_cost(x, y, _theta, _lambda),
        theta,
        method="BFGS",
        options={"maxiter": 500, "disp": False},
    )
    return np.array([result.x])
