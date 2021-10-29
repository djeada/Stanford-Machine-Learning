""""
The goal of this module is to implement all algorithms and numerical
methods needed to solve the Task 3 from the coding homeworks in the
Machine Learning course on coursera.com.
"""

import numpy as np
from scipy import optimize
from typing import Callable


def hypothesis_function(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Implementation of the logistic hypothesis function.

    Args:
      x:
        Input dataset. A matrix with m rows and n columns.
      theta:
        The parameters for the regression function. An n+1-element vector.

    Returns:
        A vector containing the values of the sigmoid function obtained after
        the function was given the dot product of the matrix x and the vector theta.
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

    return sigmoid(np.dot(x, theta))


def compute_cost(
    x: np.ndarray, y: np.ndarray, theta: np.ndarray = None, _lambda: int = 0
) -> np.float64:
    """
    Calculate the cost of logistic regression. Calculates the cost of using theta as
    the logistic regression parameter for the fit of the given data points.

    Args:
      x:
        Input dataset. A matrix with m rows and n columns.
      y:
        The values corresponding to the input dataset. An m-element vector.
      theta:
        The parameters for the regression function. An n+1-element vector.
     _lambda:
        For regularization, non-zero lambda values are utilized.
        There is no regularization when lambda is set to 0.

    Returns:
      The cost of regression function.
    """
    if theta is None:
        theta = np.zeros((x.shape[1], 1)).reshape(-1)

    m = y.size

    a = np.dot(-np.array(y).T, np.log(hypothesis_function(x, theta)))
    b = np.dot((1 - np.array(y)).T, np.log(1 - hypothesis_function(x, theta)))
    j = (1.0 / m) * np.sum(a - b)
    reg = (_lambda / 2 * m) * np.sum(np.dot(theta[1:].T, theta[1:]))

    return j + reg


def compute_gradient(
    x: np.ndarray, y: np.ndarray, theta: np.ndarray = None, _lambda: int = 0
) -> np.ndarray:
    """
     Compute the gradient by taking the partial derivatives of the cost for each parameter in theta.

     Args:
      x:
        Input dataset. A matrix with m rows and n columns.
      y:
        The values corresponding to the input dataset. An m-element vector.
      theta:
        The parameters for the regression function. An n+1-element vector.
      _lambda:
        For regularization, non-zero lambda values are utilized.
        There is no regularization when lambda is set to 0.

    Returns:
      Computed gradient.
    """
    if theta is None:
        theta = np.zeros((x.shape[1], 1)).reshape(-1)

    m = y.size
    grad = 1 / m * np.dot(x.T, hypothesis_function(x, theta) - y.T)
    grad[1:] += theta[1:] * (_lambda / m)

    return grad


def optimize_theta(
    x: np.ndarray, y: np.ndarray, theta: np.ndarray = None, _lambda: int = 0
) -> np.ndarray:
    """
    Calculate the optimal value of theta. The derivative terms need to be specified explicitly when  using the "fmin_cg"
    function as opposed to "fmin" function. We supply it with our compute_gradient function.

    Args:
      x:
        Input dataset. A matrix with m rows and n columns.
      y:
        The values corresponding to the input dataset. An m-element vector.
      theta:
       The parameters for the regression function. An n+1-element vector.
      _lambda:
       For regularization, non-zero lambda values are utilized.
       There is no regularization when lambda is set to 0.

    Returns:
     The optimized value of theta.
    """
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
    """
    Train the logistic regression classifiers. Each digit has theta vector
    associated with it.

     Args:
      x:
        Input dataset. A matrix with m rows and n columns.
      y:
        The values corresponding to the input dataset. An m-element vector.

    Returns:
        Matrix consisting of theta vectors.
    """
    thetas = list()

    for digit in range(1, 11):
        print(f"Optimizing digit {digit % 10}")
        thetas.append(
            optimize_theta(x, np.array([1 if elem[0] == digit else 0 for elem in y]))
        )

    return np.array(thetas)


def predict_one_vs_all(row: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    """
    Make a logistic regression prediction for a single example (row) from the matrix X.

    Args:
      row:
        A single row form the matrix X.
      thetas:
        A matrix of theta vectors. Each theta represents a different digit's parameters.

    Returns:
      A prediction vector represents the predicted label for a given row.
    """
    hypothesis = np.array([hypothesis_function(row, theta) for theta in thetas])
    return np.argmax(hypothesis)


def calculate_accuracy(
    x: np.ndarray,
    y: np.ndarray,
    theta: tuple[np.ndarray, np.ndarray],
    prediction_function: Callable,
) -> float:
    """
    Calculate the proportion of properly classified samples.

    Args:
      x:
        Input dataset. A matrix with m rows and n columns.
      y:
        The values corresponding to the input dataset. An m-element vector.
      theta:
       The parameters for the regression function. An n+1-element vector.
      prediction_function:
        A function used to make the prediction.

    Returns:
      A float representing the proportion of properly classified samples.
    """
    n = x.shape[0]
    correct = 0

    for i in range(n):
        prediction = prediction_function(x[i], theta) + 1
        if prediction == y[i]:
            correct += 1

    return correct / n


def propagate_forward(row: np.ndarray, thetas: np.ndarray) -> np.ndarray:
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
      An array representing output layer of the neural network.
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

    hidden_layer = sigmoid(np.dot(row, theta_1.T))
    hidden_layer = np.concatenate([np.ones([1]), hidden_layer])

    return sigmoid(np.dot(hidden_layer, theta_2.T))


def predict_nn(row: np.ndarray, thetas: np.ndarray) -> np.ndarray:
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
    return np.argmax(propagate_forward(row, thetas))
