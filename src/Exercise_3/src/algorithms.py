""""
The goal of this module is to implement all algorithms and numerical
methods needed to solve the Task 3 from the coding homeworks in the
Machine Learning course on coursera.com.
"""

import numpy as np
from scipy import optimize
from typing import Callable


def hypothesis_function(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    def sigmoid(z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    return sigmoid(np.dot(x, theta))


def compute_cost(x: np.ndarray, y: np.ndarray, theta: np.ndarray = None, _lambda: int = 0) -> np.float64:
    if theta is None:
        theta = np.zeros((x.shape[1], 1)).reshape(-1)

    m = y.size

    a = np.log(hypothesis_function(x, theta)).dot(-y.T)
    b = np.log(1 - hypothesis_function(x, theta)).dot(1 - y.T)
    return 1 / (2 * m) * ((a - b) * 2 + theta.T.dot(theta) * _lambda)


def compute_gradient(x: np.ndarray, y: np.ndarray, theta: np.ndarray = None, _lambda: int = 0) -> np.ndarray:
    if theta is None:
        theta = np.zeros((x.shape[1], 1)).reshape(-1)

    m = y.size
    grad = 1 / m * np.dot(x.T, hypothesis_function(x, theta) - y.T)
    grad[1:] += theta[1:] * (_lambda / m)

    return grad


def optimize_theta(x: np.ndarray, y: np.ndarray, theta: np.ndarray = None, _lambda: int = 0) -> np.ndarray:
    if theta is None:
        theta = np.zeros((x.shape[1], 1)).reshape(-1)

    return optimize.fmin_cg(
        lambda _theta: compute_cost(x, y, _theta, _lambda),
        x0=theta,
        fprime=lambda _theta: compute_gradient(x, y, _theta, _lambda),
        maxiter=50,
        disp=False,
        full_output=True,
    )[0]


def train_model(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    theta = list()

    for digit in range(1, 11):
        print(f"Optimizing digit {digit % 10}")
        theta.append(
            optimize_theta(x, np.array([1 if elem[0] == digit else 0 for elem in y]))
        )

    return np.array(theta)


def predict_one_vs_all(row: np.ndarray, theta: np.ndarray) -> np.int64:
    hypothesis = np.array([hypothesis_function(row, theta[i]) for i in range(10)])
    return np.argmax(hypothesis)


def calculate_training_accuracy(x: np.ndarray, y: np.ndarray, theta: np.ndarray,
                                prediction_function: Callable) -> float:
    n = x.shape[0]
    correct = 0

    for i in range(n):
        prediction = prediction_function(x[i], theta) + 1
        if prediction == y[i]:
            correct += 1

    return correct / n


def propagate_forward(row: np.ndarray, theta: np.ndarray) -> np.ndarray:
    def sigmoid(z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    theta_1, theta_2 = theta

    hidden_layer = sigmoid(np.dot(row, theta_1.T))
    hidden_layer = np.concatenate([np.ones([1]), hidden_layer])

    return sigmoid(np.dot(hidden_layer, theta_2.T))


def predict_nn(row: np.ndarray, theta: np.ndarray) -> np.int64:
    return np.argmax(propagate_forward(row, theta))
