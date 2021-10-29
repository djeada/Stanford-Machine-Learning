""""
The goal of this module is to implement all algorithms and numerical
methods needed to solve the Task 5 from the coding homeworks in the
Machine Learning course on coursera.com.
"""

from typing import Tuple
import numpy as np
import scipy.optimize


def hypothesis_function(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Hypothesis function for linear regression. It is a linear function  of the form: 
    h(x) = theta0 + theta1 * x1 + theta2 * x2 + ... + thetaN * xN.  
    Theta is a vector containing the parameter values. 
    
    Args:
        x: Matrix of features.
        theta: Vector of parameters.
    
    Returns:
        Vector of predictions.
    """
    return np.dot(x, theta)


def compute_cost(
    x: np.ndarray, y: np.ndarray, theta: np.ndarray = None, _lambda: int = 0
) -> np.float64:
    """
    Computes the cost of using theta as the parameter for linear regression to fit the data points in x and y. 
    
    Args:
        x: Matrix of features.
        y: Vector of labels.
        theta: Vector of parameters.
        _lambda: Regularization parameter.
    
    Returns:
        Cost of using theta as the parameter for linear regression to fit the data points in x and y.
    """
    if theta is None:
        theta = np.zeros((x.shape[1], 1))

    m = y.size

    j = (
        1
        / (2 * m)
        * np.dot(
            (hypothesis_function(x, theta).reshape((m, 1)) - y).T,
            (hypothesis_function(x, theta).reshape((m, 1)) - y),
        )
    )
    reg = _lambda / (2 * m) * np.dot(theta[1:].T, theta[1:])

    return (j + reg)[0][0]


def compute_gradient(
    x: np.ndarray, y: np.ndarray, theta: np.ndarray = None, _lambda: int = 0
) -> np.ndarray:
    """
    Computes the gradient of the cost function.

    Args:
        x: Matrix of features.
        y: Vector of labels.
        theta: Vector of parameters.
        _lambda: Regularization parameter.
    
    Returns:
        Vector of gradient.
    """
    if theta is None:
        theta = np.zeros((x.shape[1], 1))

    m = y.size
    theta = theta.reshape((theta.shape[0], 1))

    gradient = 1 / m * np.dot(x.T, hypothesis_function(x, theta) - y)
    reg = _lambda / m * theta

    # don't regularize the bias term
    reg[0] = 0

    return gradient + reg.reshape((gradient.shape[0], 1))


def optimize_theta(
    x: np.ndarray, y: np.ndarray, theta: np.ndarray, _lambda: int = 0
) -> np.ndarray:
    """
    Optimizes theta using the scipy.optimize.minimize function.

    Args:
        x: Matrix of features.
        y: Vector of labels.
        theta: Vector of parameters.
        _lambda: Regularization parameter.
    
    Returns:
        Vector of optimized parameters.
    """
    return scipy.optimize.minimize(
        lambda _theta: compute_cost(x, y, _theta, _lambda),
        x0=theta,
        method="BFGS",
        options={
            "disp": False,
            "gtol": 1e-05,
            "eps": 1.4901161193847656e-08,
            "return_all": False,
            "maxiter": None,
        },
    ).x


def construct_polynomial_matrix(x: np.ndarray, p: int) -> np.ndarray:
    """
    Takes an x matrix and returns an x matrix with additional columns.
    First additional column is 2'nd column with all values squared,
    the next additional column is 2'nd column with all values cubed etc.    

    Args:
        x: Matrix of features.
        p: Degree of the polynomial.
    
    Returns:
        Matrix of features with additional columns.
    """

    p_matrix = x.copy()

    for i in range(2, p + 2):
        p_matrix = np.insert(p_matrix, p_matrix.shape[1], np.power(x[:, 1], i), axis=1)

    return p_matrix


def normalize_features(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalizes the features in x.  
    
    Args:
        x: Matrix of features.

    Returns:
        The normalized x, the mean of the original features and the standard deviation of the original features
    """
    x_norm = x.copy()

    feature_means = np.mean(x_norm, axis=0)
    x_norm[:, 1:] = x_norm[:, 1:] - feature_means[1:]
    feature_stds = np.std(x_norm, axis=0, ddof=1)
    x_norm[:, 1:] = x_norm[:, 1:] / feature_stds[1:]

    return x_norm, feature_means, feature_stds
