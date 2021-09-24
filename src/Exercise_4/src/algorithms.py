""""
The goal of this module is to implement all algorithms and numerical
methods needed to solve the Task 4 from the coding homeworks in the
Machine Learning course on coursera.com.
"""

import numpy as np
from scipy import optimize


def forward_propagation(x: np.ndarray, theta: tuple[np.ndarray, np.ndarray]) -> tuple:
    def sigmoid(z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    theta_1, theta_2 = theta
    m = x.shape[0]

    # input layer a_1 is row from x matrix with added column of bias units (ones)
    a_1 = x

    # hidden layer computed with dot product of input layer and theta
    # plus column of bias units
    z_2 = np.dot(a_1, theta_1.T)
    a_2 = sigmoid(z_2)
    a_2 = np.concatenate([np.ones((m, 1)), a_2], axis=1)

    # output layer, computed like the hidden layer
    z_3 = np.dot(a_2, theta_2.T)
    a_3 = sigmoid(z_3)

    return (a_1, a_2, a_3), (z_2, z_3)


def compute_cost(x: np.ndarray, y: np.ndarray, theta: tuple[np.ndarray, np.ndarray] = None,
                 _lambda: int = 0) -> np.float64:
    if theta is None:
        theta = np.zeros((x.shape[1], 1)).reshape(-1)

    theta_1, theta_2 = theta
    m = y.size

    y_matrix = np.eye(10)[y.reshape(-1)]

    j = -1 / m * (np.sum(y_matrix * np.log(x) + (1 - y_matrix) * np.log(1 - x)))
    reg = (
            _lambda
            / (2 * m)
            * (np.sum(np.square(theta_1[:, 1:])) + np.sum(np.square(theta_2[:, 1:])))
    )

    return j + reg


def backpropagation(a: tuple, z: tuple, y: np.ndarray, theta: tuple[np.ndarray, np.ndarray], _lambda: int = 0) -> tuple:
    """
    """

    def sigmoid_gradient(_z: np.ndarray) -> np.ndarray:
        """
        """
        gradient = 1 / (1 + np.exp(-_z))

        return gradient * (1 - gradient)

    a_1, a_2, a_3 = a
    z_2, z_3 = z
    theta_1, theta_2 = theta
    m = y.size

    y_matrix = np.eye(10)[y.reshape(-1)]

    d_3 = a_3 - y_matrix
    d_2 = np.dot(d_3, theta_2[:, 1:]) * sigmoid_gradient(z_2)

    theta_1_grad = 1 / m * np.dot(d_2.T, a_1)
    theta_2_grad = 1 / m * np.dot(d_3.T, a_2)

    theta_2_grad[:, 1:] += _lambda / m * theta_2[:, 1:]
    theta_1_grad[:, 1:] += _lambda / m * theta_1[:, 1:]

    return theta_1_grad.ravel(), theta_2_grad.ravel()


def random_weights(num_input_connections: int, num_output_connections: int, epsilon: float = 0.12) -> np.ndarray:
    """
    """
    return (
            np.random.rand(num_output_connections, 1 + num_input_connections) * 2 * epsilon
            - epsilon
    )


def split_theta(theta: np.ndarray, input_layer_size: int, hidden_layer_size: int, num_labels: int) -> tuple[
    np.ndarray, np.ndarray]:
    """
    """
    theta_1 = np.reshape(
        theta[: hidden_layer_size * (input_layer_size + 1)],
        (hidden_layer_size, (input_layer_size + 1)),
    )

    theta_2 = np.reshape(
        theta[(hidden_layer_size * (input_layer_size + 1)):],
        (num_labels, (hidden_layer_size + 1)),
    )

    return theta_1, theta_2


def find_optimized_theta(
        x: np.ndarray,
        y: np.ndarray,
        theta: np.ndarray,
        input_layer_size: int,
        hidden_layer_size: int,
        num_labels: int,
        options: dict = {"maxiter": 100},
        _lambda: int = 1,
) -> np.ndarray:
    def f_wrapper(_theta: np.ndarray) -> tuple:
        """
        """
        reshaped_theta = split_theta(
            _theta, input_layer_size, hidden_layer_size, num_labels
        )
        a, z = forward_propagation(x, reshaped_theta)
        cost = compute_cost(a[-1], y, reshaped_theta)
        gradient = backpropagation(a, z, y, reshaped_theta)
        return cost, np.concatenate([gradient[0].ravel(), gradient[1].ravel()])

    return optimize.minimize(
        f_wrapper, theta, jac=True, method="TNC", options=options
    ).x


def calculate_training_accuracy(x: np.ndarray, y: np.ndarray, theta: tuple[np.ndarray, np.ndarray]) -> float:
    """
    """
    n = x.shape[0]
    correct = 0

    predictions = predict(x, theta)

    for prediction, expected in zip(predictions, y):
        if prediction == expected:
            correct += 1

    return correct / n


def predict(x: np.ndarray, theta: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    def sigmoid(z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    theta_1, theta_2 = theta
    m = x.shape[0]

    h_1 = sigmoid(np.dot(np.concatenate([np.ones((m, 1)), x], axis=1), theta_1.T))
    h_2 = sigmoid(np.dot(np.concatenate([np.ones((m, 1)), h_1], axis=1), theta_2.T))

    return np.argmax(h_2, axis=1)
