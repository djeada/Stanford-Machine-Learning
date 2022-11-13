""""
The goal of this module is to implement all algorithms and numerical
methods needed to solve the Task 4 from the coding homeworks in the
Machine Learning course on coursera.com.
"""
from typing import Tuple

import numpy as np
from scipy import optimize


def propagate_forward(
    row: np.ndarray, thetas: Tuple[np.ndarray, np.ndarray]
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Implementation of the forward propagation algorithm. The function is
    a simple neural network including input layer, single hidden layer
    and output.

    Args:
      row:
        A single row form the matrix X.
      thetas:
        Two sets of weights used for neural network training.

    Returns:
      A tuple of arrays made up of an input layer, a hidden layer, and an output layer.
    """

    def sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Sigmoid function implementation. It will apply the sigmoid function
        to each member of a given vector z.

        Args:
         z:
           An n+1-element vector.

        Returns:
           A vector of sigmoid function values for every element of z.
        """
        return 1 / (1 + np.exp(-z))

    theta_1, theta_2 = thetas
    m = row.shape[0]

    # input layer a_1 is row from matrix X with added column of bias units (ones)
    a_1 = row

    # hidden layer computed with dot product of input layer and theta
    # plus column of bias units
    z_2 = np.dot(a_1, theta_1.T)
    a_2 = sigmoid(z_2)
    a_2 = np.concatenate([np.ones((m, 1)), a_2], axis=1)

    # output layer, computed like the hidden layer
    z_3 = np.dot(a_2, theta_2.T)
    a_3 = sigmoid(z_3)

    return (a_1, a_2, a_3), (z_2, z_3)


def compute_cost(
    x: np.ndarray,
    y: np.ndarray,
    thetas: Tuple[np.ndarray, np.ndarray] = None,
    _lambda: int = 0,
) -> np.float64:
    """
    Calculate the cost of a simple neural network. Calculates the cost of using theta as
    the neural network parameter for the fit of the given data points.

    Args:
      x:
        Input dataset. A matrix with m rows and n columns.
      y:
        The values corresponding to the input dataset. An m-element vector.
      thetas:
        Two sets of weights used for neural network training.
     _lambda:
        For regularization, non-zero lambda values are utilized.
        There is no regularization when lambda is set to 0.

    Returns:
      The cost of neural network.
    """
    if thetas is None:
        thetas = np.zeros((x.shape[1], 1)).reshape(-1)

    theta_1, theta_2 = thetas
    m = y.size

    y_matrix = np.eye(10)[y.reshape(-1)]

    j = -1 / m * (np.sum(y_matrix * np.log(x) + (1 - y_matrix) * np.log(1 - x)))
    reg = (
        _lambda
        / (2 * m)
        * (np.sum(np.square(theta_1[:, 1:])) + np.sum(np.square(theta_2[:, 1:])))
    )

    return j + reg


def backpropagation(
    a: Tuple[np.ndarray, np.ndarray, np.ndarray],
    z: Tuple[np.ndarray, np.ndarray],
    y: np.ndarray,
    thetas: Tuple[np.ndarray, np.ndarray],
    _lambda: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implementation of backpropagation algorithm for a simple neural network. Computes the optimal weights
    for given values of neural network layers.

    Args:
      a:
        A tuple of arrays made up of an input layer, a hidden layer, and an output layer.
      z:
        A tuple of arrays made up of an input layer, a hidden layer, and an output layer.
      y:
        The values corresponding to the input dataset. An m-element vector.
      thetas:
        Two sets of weights used for neural network training.
      _lambda:


    Returns:
      Optimal weights for the neural network.
    """

    def sigmoid_gradient(_z: np.ndarray) -> np.ndarray:
        """
        Implementation of sigmoid function gradient. It will apply the sigmoid function
        gradient to each member of a given vector z.

        Args:
          _z:
            An n+1-element vector.

        Returns:
            A vector of sigmoid function gradient values for every element of z.
        """
        gradient = 1 / (1 + np.exp(-_z))

        return gradient * (1 - gradient)

    a_1, a_2, a_3 = a
    z_2, _ = z
    theta_1, theta_2 = thetas
    m = y.size

    y_matrix = np.eye(10)[y.reshape(-1)]

    d_3 = a_3 - y_matrix
    d_2 = np.dot(d_3, theta_2[:, 1:]) * sigmoid_gradient(z_2)

    theta_1_grad = 1 / m * np.dot(d_2.T, a_1)
    theta_2_grad = 1 / m * np.dot(d_3.T, a_2)

    theta_2_grad[:, 1:] += _lambda / m * theta_2[:, 1:]
    theta_1_grad[:, 1:] += _lambda / m * theta_1[:, 1:]

    return theta_1_grad.ravel(), theta_2_grad.ravel()


def random_weights(
    input_layer_size: int, hidden_layer_size: int, epsilon: float = 0.12
) -> np.ndarray:
    """
    Randomly initialize the weights of a layer in a neural network.
    
    Args:
      input_layer_size:
        The number of nodes in the input layer.
      hidden_layer_size:
        The number of nodes in the hidden layer.
      epsilon:
        The epsilon value used to generate random weights.

    Returns:
      An array of random weights.
    """
    return (
        np.random.rand(hidden_layer_size, 1 + input_layer_size) * 2 * epsilon - epsilon
    )


def split_theta(
    theta: np.ndarray, input_layer_size: int, hidden_layer_size: int, num_labels: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split theta values into two sets of weights used in the neural network.
    
    Args:
      theta:
        The parameters for the neural network. An n+1-element vector.
      input_layer_size:
        The number of nodes in the input layer.
      hidden_layer_size:
        The number of nodes in the hidden layer.
      num_labels:
        The number of nodes in the output layer.


    Returns:
      Two sets of weights.
    """
    theta_1 = np.reshape(
        theta[: hidden_layer_size * (input_layer_size + 1)],
        (hidden_layer_size, (input_layer_size + 1)),
    )

    theta_2 = np.reshape(
        theta[(hidden_layer_size * (input_layer_size + 1)) :],
        (num_labels, (hidden_layer_size + 1)),
    )

    return theta_1, theta_2


def optimize_theta(
    x: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    input_layer_size: int,
    hidden_layer_size: int,
    num_labels: int,
    options: dict = {"maxiter": 100},
) -> np.ndarray:
    """
    Calculate the optimal value of theta. The derivative terms need to be specified explicitly when using the "minimize"
    function. We supply it with our compute_gradient function.

    Args:
      x:
        Input dataset. A matrix with m rows and n columns.
      y:
        The values corresponding to the input dataset. An m-element vector.
      theta:
        The parameters for the neural network. An n+1-element vector.
      input_layer_size:
        The number of nodes in the input layer.
      hidden_layer_size:
        The number of nodes in the hidden layer.
      num_labels:
        The number of nodes in the output layer.
      options:
        Options for the minimization algorithm.

    Returns:
     The optimized value of theta.
    """

    def f_wrapper(_theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
          _theta:
            The parameters for the neural network. An n+1-element vector.

        Returns:
          A prediction vector represents the predicted label for a given row.
        """
        reshaped_theta = split_theta(
            _theta, input_layer_size, hidden_layer_size, num_labels
        )
        a, z = propagate_forward(x, reshaped_theta)
        cost = compute_cost(a[-1], y, reshaped_theta)
        gradient = backpropagation(a, z, y, reshaped_theta)
        return cost, np.concatenate([gradient[0].ravel(), gradient[1].ravel()])

    return optimize.minimize(
        f_wrapper, theta, jac=True, method="TNC", options=options
    ).x


def calculate_accuracy(
    x: np.ndarray, y: np.ndarray, thetas: Tuple[np.ndarray, np.ndarray]
) -> float:
    """
    Calculate the proportion of properly classified samples.

    Args:
      x:
        Input dataset. A matrix with m rows and n columns.
      y:
        The values corresponding to the input dataset. An m-element vector.
      thetas:
        Two sets of weights used for neural network training.

    Returns:
      A float representing the proportion of properly classified samples.
    """
    n = x.shape[0]
    correct = 0

    predictions = predict(x, thetas)

    for prediction, expected in zip(predictions, y):
        if prediction == expected:
            correct += 1

    return correct / n


def predict(row: np.ndarray, thetas: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Using a trained neural network, make a prediction for a single example (row) from the matrix X.

    Args:
      row:
        A single row form the matrix X.
      thetas:
        Two sets of weights used for neural network training.

    Returns:
      A prediction vector represents the predicted label for a given row.
    """

    def sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Sigmoid function implementation. It will apply the sigmoid function
        to each member of a given vector z.

        Args:
         z:
           An n+1-element vector.

        Returns:
           A vector of sigmoid function values for every element of z.
        """
        return 1 / (1 + np.exp(-z))

    theta_1, theta_2 = thetas
    m = row.shape[0]

    h_1 = sigmoid(np.dot(np.concatenate([np.ones((m, 1)), row], axis=1), theta_1.T))
    h_2 = sigmoid(np.dot(np.concatenate([np.ones((m, 1)), h_1], axis=1), theta_2.T))

    return np.argmax(h_2, axis=1)
