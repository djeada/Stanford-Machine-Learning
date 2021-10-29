""""
The goal of this module is to implement all the visualization
tools needed to graph the data and results of the computations
for the Task 5 from the coding homeworks in the Machine Learning
course on coursera.com.
"""

from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

import algorithms


def plot_data(
    data: Tuple[np.ndarray, np.ndarray], title: str = "Scatter Plot Of Training Data"
) -> None:
    """
    Plot the data points on a scatter plot. The function uses the matplotlib library. 
    The plot is saved as a png file.

    Args:
      data:
        A tuple of x and y values for the points to be plotted.
      title:
        A string that serves as both the plot's title and the saved figure's filename.

    Returns:
      None
    """
    x, y = data
    plt.figure(figsize=(10, 6))
    plt.scatter(x[:, 1], y, c="red", s=50)
    plt.ylabel("Water flowing out of the dam (y)")
    plt.xlabel("Change in water level (x)")
    plt.title(title)
    plt.savefig(title.lower().replace(" ", "_"))


def plot_linear_regression_fit(
    data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    title: str = "Linear Regression Fit",
) -> None:
    """
    Plot the linear regression fit on a scatter plot. The function uses the matplotlib library.
    The plot is saved as a png file.

    Args:
      data:
        A tuple of x and y values for the points to be plotted.

      title:
        A string that serves as both the plot's title and the saved figure's filename.

    Returns:
      None
    """

    def fit(_theta, alpha):
        return _theta[0] + _theta[1] * alpha

    theta, x, y = data

    plt.figure(figsize=(10, 6))

    plt.scatter(x[:, 1], y, c="red", s=50)
    plt.plot(x[:, 1], fit(theta, x[:, 1]), "b-", label="Linear regression")
    plt.ylabel("Water flowing out of the dam (y)")
    plt.xlabel("Change in water level (x)")
    plt.legend(loc="lower right")
    plt.title(title)
    plt.savefig(title.lower().replace(" ", "_"))


def plot_learning_curve(data: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """
    Plot the learning curve for linear regression. The function uses the matplotlib library.
    The plot is saved as a png file.

    Args:
      data:
        A tuple consiting of the training set, validation set, and test set.

    Returns:
      None
    """
    training_set, validation_set, theta = data
    x, y = training_set
    x_validation, y_validation = validation_set

    m_history = list()
    train_error_history = list()
    validation_error_history = list()

    for i in range(1, 12):
        x_subset = x[:i, :]
        y_subset = y[:i]
        theta_optimized = algorithms.optimize_theta(x_subset, y_subset, theta)

        m_history.append(y_subset.shape[0])
        train_error_history.append(
            algorithms.compute_cost(x_subset, y_subset, theta_optimized)
        )
        validation_error_history.append(
            algorithms.compute_cost(x_validation, y_validation, theta_optimized)
        )

    plt.figure(figsize=(10, 6))
    plt.plot(m_history, train_error_history, label="Train")
    plt.plot(m_history, validation_error_history, label="Cross Validation")
    plt.xlabel("Number of training examples")
    plt.ylabel("Error")
    plt.title("Learning curve for linear regression")
    plt.legend()
    plt.savefig("learning_curve")


def plot_polynomial_regression_fit(
    data: Tuple[np.ndarray, np.ndarray, np.ndarray, list, list, int], n_points: int = 50
) -> None:
    """
    Plot the polynomial regression fit on a scatter plot. The function uses the matplotlib library.
    The plot is saved as a png file.

    Args:
      data:
        A tuple of x and y values for the points, theta parameters, means and standard deviations and lambda.
      n_points:
        The number of points to be plotted on the polynomial regression fit.

    Returns:
      None
    """
    theta, x, y, means, stds, _lambda = data

    x_range = np.linspace(-50, 50, n_points)

    x_temp = np.ones((len(x_range), 1))
    x_temp = np.insert(x_temp, x_temp.shape[1], x_range.T, axis=1)
    x_temp = algorithms.construct_polynomial_matrix(x_temp, len(theta) - 2)

    x_temp[:, 1:] -= means[1:]
    x_temp[:, 1:] /= stds[1:]

    plt.figure(figsize=(10, 6))
    plt.scatter(x[:, 1], y, c="red", s=50)
    plt.plot(
        x_range,
        algorithms.hypothesis_function(x_temp, theta),
        "b--",
        label="Polynomial regression",
    )
    plt.ylabel("Water flowing out of the dam (y)")
    plt.xlabel("Change in water level (x)")
    plt.legend(loc="lower right")
    plt.title(f"polynomial regression fit (lambda={_lambda})")
    plt.savefig(f"polynomial_regression_fit_{_lambda}")


def plot_polynomial_learning_curve(
    data: Tuple[np.ndarray, np.ndarray, np.ndarray], _lambda: int = 0, p: int = 5
) -> None:
    """
    Plot the learning curve for polynomial regression. The function uses the matplotlib library.
    The plot is saved as a png file.

    Args:
      data:
        A tuple consiting of the training set, validation set, and test set.
      _lambda:
        The regularization parameter.
      p:
        The degree of the polynomial.

    Returns:
      None
    """
    training_set, validation_set, theta = data
    x, y = training_set
    x_validation, y_validation = validation_set
    x_validation, _, __ = algorithms.normalize_features(
        algorithms.construct_polynomial_matrix(x_validation, p)
    )

    m_history = list()
    train_error_history = list()
    validation_error_history = list()

    for i in range(1, 12):
        x_subset = x[:i, :]
        x_subset = algorithms.construct_polynomial_matrix(x_subset, p)
        x_subset, _, __ = algorithms.normalize_features(x_subset)
        y_subset = y[:i]
        theta_optimized = algorithms.optimize_theta(x_subset, y_subset, theta, _lambda)

        m_history.append(y_subset.shape[0])
        train_error_history.append(
            algorithms.compute_cost(x_subset, y_subset, theta_optimized, _lambda)
        )
        validation_error_history.append(
            algorithms.compute_cost(
                x_validation, y_validation, theta_optimized, _lambda
            )
        )

    plt.figure(figsize=(10, 6))
    plt.plot(m_history, train_error_history, label="Train")
    plt.plot(m_history, validation_error_history, label="Cross Validation")
    plt.xlabel("Number of training examples")
    plt.ylabel("Error")
    plt.legend()
    plt.title(f"Learning curve for polynomial regression (lambda={_lambda})")
    plt.savefig(f"polynomial_learning_curve_lambda_{_lambda}")
