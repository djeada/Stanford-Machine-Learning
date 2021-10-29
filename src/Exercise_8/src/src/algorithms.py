""""
The goal of this module is to implement all algorithms and numerical
methods needed to solve the Task 8 from the coding homeworks in the
Machine Learning course on coursera.com.
"""

import numpy as np
import scipy.stats as stats
from scipy import optimize


def compute_f1(predictions: np.ndarray, real_values: np.ndarray) -> float:
    """
    Compute the F1 score for the predictions. Equation of F1 has the form: 
    F1 = 2 * (precision * recall) / (precision + recall)
    The definition of precision is the number of correct predictions divided by the total number of predictions.
    The definition of recall is the number of correct predictions divided by the total number of real values.

    Args:
        predictions: array of predictions
        real_values: array of real values

    Returns:
      The F1 score for the predictions.        
    """

    p, r = 0, 0
    if np.sum(predictions) != 0:
        p = np.sum(
            [real_values[x] for x in range(predictions.shape[0]) if predictions[x]]
        ) / np.sum(predictions)
    if np.sum(real_values) != 0:
        r = np.sum(
            [predictions[x] for x in range(real_values.shape[0]) if real_values[x]]
        ) / np.sum(real_values)

    return 2 * p * r / (p + r) if (p + r) else 0.0


def get_gaussian_parameters(x: np.ndarray) -> tuple:
    """
    Compute the mean and the variance of the data in x.

    Args:
        x: array of data
    
    Returns:
        A tuple consisting of:
        - mu: mean of the data
        - sigma_2: variance of the data
    """
    mu, sigma_2 = np.mean(x, axis=0), np.var(x, axis=0)
    return mu, sigma_2


def compute_gauss(grid: np.ndarray, mu: np.ndarray, sigma_2: np.ndarray) -> np.ndarray:
    """
    Compute the gaussian function for the given grid.

    Args:
        grid: array of grid values
        mu: mean of the data
        sigma_2: variance of the data
    
    Returns:
        The gaussian function for the given grid.
    """
    return stats.multivariate_normal.pdf(x=grid, mean=mu, cov=sigma_2)


def select_threshold(
    y_cv: np.ndarray, cv_set: np.ndarray, n_steps: int = 1000
) -> tuple:
    """
    Function to select the best epsilon value from the cross validation set. 
    Loops over the values of epsilon and computes the F1 score for each value.

    Args:
        y_cv: array of cross validation values
        cv_set: array of cross validation set
        n_steps: number of steps to compute the F1 score
    
    Returns:
        A tuple consisting of:
        - best_epsilon: the best epsilon value
        - best_f1: the best F1 score
    """
    epsilons = np.linspace(np.min(cv_set), np.max(cv_set), n_steps)

    best_f1, best_eps = 0, 0
    real_values = (y_cv == 1).flatten()
    for epsilon in epsilons:
        predictions = cv_set < epsilon
        f1 = compute_f1(predictions, real_values)
        if f1 > best_f1:
            best_f1 = f1
            best_eps = epsilon

    return best_f1, best_eps


def collaborative_filtering_cost(
    x: np.ndarray, theta: np.ndarray, y: np.ndarray, r: np.ndarray, _lambda: float = 1.0
):
    """
    Collaborative filtering cost function. The cost function is the sum of the squared errors of the predictions.

    Args:
        x: array of data
        theta: array of parameters
        y: array of ratings
        r: array of binary ratings
        _lambda: regularization parameter

    Returns:
        The cost of the predictions.
    """

    j = np.sum(np.sum(((x @ theta.T - y) * r) ** 2)) / 2

    # Theta regularization
    j += np.sum(np.sum(theta ** 2)) * (_lambda / 2)
    # X regularization
    j += np.sum(np.sum(x ** 2)) * (_lambda / 2)

    return j


def collaborative_filtering_gradient(
    x: np.ndarray, theta: np.ndarray, y: np.ndarray, r: np.ndarray, _lambda: float = 1.0
) -> np.ndarray:
    """
    Collaborative filtering gradient function. The gradient is the sum of the errors of the predictions.

    Args:
        x: array of data
        theta: array of parameters
        y: array of ratings
        r: array of binary ratings
        _lambda: regularization parameter
    
    Returns:
        The gradient of the predictions.
    """

    # compute gradient
    x_grad = ((x @ theta.T - y) * r) @ theta
    theta_grad = ((x @ theta.T - y) * r).T @ x

    # regularization
    x_grad += _lambda * x_grad
    theta_grad += _lambda * theta_grad

    return np.hstack((x_grad.flatten(), theta_grad.flatten()))


def normalize_ratings(y: np.ndarray, r: np.ndarray) -> tuple:
    """
    Normalize the ratings. The normalization is done by dividing the ratings by the number of ratings.

    Args:
        y: array of ratings
        r: array of binary ratings
    
    Returns:
        A tuple consisting of:
        - y_norm: normalized ratings
        - r_norm: normalized binary ratings
    """

    mean = np.sum(y, axis=1) / np.sum(r, axis=1)
    mean = mean.reshape((mean.shape[0], 1))

    return y - mean, mean


def update_matrices_with_new_ratings(y: np.ndarray, r: np.ndarray) -> tuple:
    """
    Insert new ratings into the Y matrix and the corresponding row into the R matrix.

    Args:
        y: array of ratings
        r: array of binary ratings
    
    Returns:
        A tuple consisting of:
        - y_new: new ratings
        - r_new: new binary ratings
    """

    new_ratings = np.zeros((y.shape[0], 1))
    new_ratings[0] = 4
    new_ratings[97] = 2
    new_ratings[6] = 3
    new_ratings[11] = 5
    new_ratings[53] = 4
    new_ratings[63] = 5
    new_ratings[65] = 3
    new_ratings[68] = 5
    new_ratings[182] = 4
    new_ratings[225] = 5
    new_ratings[354] = 5

    y = np.hstack((y, new_ratings))
    r = np.hstack((r, new_ratings > 0))

    return new_ratings, y, r


def optimize_theta(
    y: np.ndarray, r: np.ndarray, params=None, _lambda: int = 0
) -> np.ndarray:
    """
    Optimize the parameters of the model. The optimization is done by minimizing the cost function.

    Args:
        y: array of ratings
        r: array of binary ratings
        params: array of parameters
        _lambda: regularization parameter
    
    Returns:
        The optimized parameters.
    """
    if params is None:
        x = np.random.rand(5, 3)
        theta = np.random.rand(4, 3)
        params = np.concatenate((x.flatten(), theta.flatten()))

    def collaborative_filtering_cost_wrapper(_params: np.ndarray) -> np.ndarray:
        """
        Wrapper for the cost function.

        Args:
            _params: array of parameters
        
        Returns:
            The cost of the predictions.
        """
        _x = _params[: 5 * 3].reshape((5, 3))
        _theta = _params[5 * 3 :].reshape((4, 3))
        return collaborative_filtering_cost(_x, _theta, y, r, _lambda)

    def collaborative_filtering_gradient_wrapper(_params: np.ndarray) -> np.ndarray:
        """
        Wrapper for the gradient function.

        Args:
            _params: array of parameters
        
        Returns:
            The gradient of the predictions.
        """
        _x = _params[: 5 * 3].reshape((5, 3))
        _theta = _params[5 * 3 :].reshape((4, 3))
        return collaborative_filtering_gradient(_x, _theta, y, r, _lambda)

    return optimize.fmin_cg(
        collaborative_filtering_cost_wrapper,
        x0=params,
        fprime=collaborative_filtering_gradient_wrapper,
        maxiter=400,
        full_output=True,
    )[0]
