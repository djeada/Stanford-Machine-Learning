""""
The goal of this module is to implement all algorithms and numerical
methods needed to solve the Task 2 from the coding homeworks in the
Machine Learning course on coursera.com.
"""

import numpy as np
from scipy import optimize


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


def optimize_theta(
    x: np.ndarray, y: np.ndarray, theta: np.ndarray = None, _lambda: int = 0
) -> np.ndarray:
    """
    Calculate the optimal value of theta. The derivative terms do not need to be specified explicitly when
    using the "fmin" function. It simply requires the cost function and uses the downhill simplex algorithm
    to minimize it.

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
        theta = np.zeros((x.shape[1], 1))

    return optimize.fmin(
        lambda _theta: compute_cost(x, y, _theta, _lambda),
        x0=theta,
        maxiter=400,
        full_output=True,
    )[0]


def predict(theta: np.ndarray, score_1: int, score_2: int) -> np.ndarray:
    """
    Using the hypothesis function, make a prediction. The logistic function
    requires an n+1-element vector and three values to be used as parameters.
    The three parameters are: 1, score 1, and score 2.

    Args:
     theta:
       An n+1-element vector.
     score_1:
       A parameter for logistic hypothesis function calculation.
     score_2:
       A parameter for logistic hypothesis function calculation.

    Returns:
     The value returned by the hypothesis function.
    """
    return hypothesis_function(theta, np.array([1, score_1, score_2]))


def calculate_accuracy(
    theta: np.ndarray, positive_points: np.ndarray, negative_points: np.ndarray
) -> float:
    """
    Calculate the proportion of properly classified samples.

    Args:
     theta:
       An array of theta values.
     positive_points:
       An array of positive points.
     negative_points:
       An array of negative points.

    Returns:
      A float representing the proportion of properly classified samples.
    """
    total = 0

    for score in positive_points:
        if predict(theta, score[1], score[2]) >= 0.5:
            total += 1

    for score in negative_points:
        if predict(theta, score[1], score[2]) < 0.5:
            total += 1

    return total / (len(positive_points) + len(negative_points))


def map_feature(
    x1_col: np.ndarray, x2_col: np.ndarray, n_degrees: int = 6
) -> np.ndarray:
    """
    The two input features are mapped to quadratic features for use in the
    regularization exercise.

    Args:
     x1_col:
       An m-element array, containing one feature for all examples.
     x2_col:
       An m-element array, containing second feature for all examples.
     n_degrees:
      An int representing the polynomial degree.

    Returns:
      A matrix of of m rows, and n_degrees columns.
    """
    result = np.ones((x1_col.shape[0], 1))

    for i in range(n_degrees):
        for j in range(i + 2):
            a = x1_col ** (i - j + 1)
            b = x2_col ** j
            result = np.hstack((result, (a * b).reshape(a.shape[0], 1)))
    return result


def optimize_regularized_theta(
    x: np.ndarray, y: np.ndarray, theta: np.ndarray = None, _lambda: int = 0
) -> np.ndarray:
    """
    Compute the optimized values of regularized theta.

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
      An array of optimized values of regularized theta.
    """
    if theta is None:
        theta = np.zeros((x.shape[1], 1))

    result = optimize.minimize(
        lambda _theta: compute_cost(x, y, _theta, _lambda),
        theta,
        method="BFGS",
        options={"maxiter": 500, "disp": False},
    )
    return np.array([result.x])
