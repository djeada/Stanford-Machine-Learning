""""
The goal of this module is to implement all algorithms and numerical
methods needed to solve the Task 1 from the coding homeworks in the
Machine Learning course on coursera.com.
"""
from typing import Tuple

import numpy as np


def hypothesis_function(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Calculate the hypothesis function. The hypothesis function is the linear regression function.

    Args:
      x:
        Input dataset. A matrix with m rows and n columns.
      theta:
        The parameters for the regression function. An n+1-element vector.

    Returns:
        A dot product of matrix x and vector theta.
    """
    return np.dot(x, theta)


def compute_cost(x: np.ndarray, y: np.ndarray, theta: np.ndarray = None) -> np.float64:
    """
    Computes the cost of using theta as the parameter for linear regression to fit the data points in x and y.
    The cost function is the sum of the squares of the differences between the predicted and actual values.
    The cost function is minimized by gradient descent.

    Args:
      x:
        Input dataset. A matrix with m rows and n columns.
      y:
        The values corresponding to the input dataset. An m-element vector.
      theta:
        The parameters for the regression function. An n+1-element vector.

    Returns:
        The cost of using theta as the parameter for linear regression to fit the data points in x and y.
    """
    if theta is None:
        theta = np.zeros((x.shape[1], 1))

    m = y.size
    a = (hypothesis_function(x, theta) - y).T
    b = hypothesis_function(x, theta) - y
    j = (1.0 / (2 * m) * np.dot(a, b))[0][0]
    return j


def gradient_descent(
    x: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray = None,
    num_iter: int = 2000,
    alpha: float = 0.01,
) -> Tuple[list, list]:
    """
    Performs gradient descent to learn theta. The function returns the theta and the cost history. 
    Theta is a vector of parameters for the regression function. The cost history is a list of the 
    cost values at each iteration.  The function will not return anything if the number of iterations 
    is less than 1. The function will not return anything if the learning rate is less than 0. 

    Args:
      x:
        Input dataset. A matrix with m rows and n columns.
      y:
        The values corresponding to the input dataset. An m-element vector.
      theta:
        The parameters for the regression function. An n+1-element vector.
      num_iter:
        The number of steps in gradient descent algorithm.
      alpha:
        The learning rate.

    Returns:
      A tuple consisting of:
      - A list of theta values after each iteration.
      - A list of the cost function values after each iteration.
    """
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


def normalize_features(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    A preprocessing step that is typically performed when dealing with learning algorithms.
    The features are normalized by substracting the mean and dividing by the standard deviation.

    Args:
      x:
        Input dataset. A matrix with m rows and n columns.

    Returns:
      A tuple consisting of:
      - A vector of means.
      - A vector of standard deviations.
    """
    means = [np.mean(x[:, 0])]
    stds = [np.std(x[:, 0])]

    for i in range(1, x.shape[1]):
        means.append(np.mean(x[:, i]))
        stds.append(np.std(x[:, i]))
        x[:, i] = (x[:, i] - means[-1]) / stds[-1]

    return np.array(means), np.array(stds)


def predict_using_normalized_features(
    means: np.ndarray,
    stds: np.ndarray,
    theta: np.ndarray,
    house_size: int,
    num_bedrooms: int,
) -> np.ndarray:
    """
    Make a prediction of value for given house size and number of bedrooms using provided theta
    vector. Normalize the y values by substring the means and dividing by standard deviations.

    Args:
      means:
        3-element vector of feature means.
      stds:
        3-element vector of feature stds.
      theta:
        The parameters for the regression function. An n+1-element vector.
      house_size:
        A house size for which the prediction should be made.
      num_bedrooms:
        A number of bedrooms for which the prediction should be made.

    Returns:
        Predicted price of the house with the given size and number of bedrooms.
    """
    y = [house_size, num_bedrooms]
    y = [(y[i] - means[i + 1]) / stds[i + 1] for i in range(len(y))]
    y.insert(0, 1)
    y = np.array(y)

    return hypothesis_function(y, theta)


def predict_from_normal_equation(
    x: np.ndarray, y: np.ndarray, house_size: int, num_bedrooms: int
) -> np.ndarray:
    """
    Using the normal equations, computes the closed-form solution to linear regression.

    Args:
      x:
        Input dataset. A matrix with m rows and n columns.
      y:
        The values corresponding to the input dataset. An m-element vector.
      house_size:
        A house size for which the prediction should be made.
      num_bedrooms:
        A number of bedrooms for which the prediction should be made.

    Returns:
      Predicted price of the house with the given size and number of bedrooms.
    """

    def norm_eq(_x: np.ndarray, _y: np.ndarray) -> np.ndarray:
        """
        Implementation of the normal equation.

        Args:
          _x:
            Input dataset. A matrix with m rows and n columns.
          _y:
            The values corresponding to the input dataset. An m-element vector.

        Returns:
            A theta vector for given _x and _y matrices.
        """
        return np.dot(np.dot(np.linalg.inv(np.dot(_x.T, _x)), _x.T), _y)

    return hypothesis_function(np.array([1, house_size, num_bedrooms]), norm_eq(x, y))
