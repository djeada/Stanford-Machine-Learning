""""
The goal of this module is to give a comprehensive solution to Task 5
from the coding homeworks from the Machine Learning course on coursera.com.
The task is broken down into smaller parts.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import readers
import algorithms
import plots

DATA_PATH = Path("../data/data.mat")


def part_1() -> None:
    """
    Regularized linear regression:
    - Data visualization.
    - Implementation of regularized linear regression gradient.
    - Showing the bias-variance tradeoff.

    Returns:
      None
    """
    training_set, validation_set, test_set = readers.read_data(DATA_PATH)
    plots.plot_data(training_set)

    x, y = training_set
    theta = np.array([[1.0], [1.0]])
    cost = algorithms.compute_cost(x, y, theta, 3)
    gradient = algorithms.compute_gradient(x, y, theta, 1)

    theta = algorithms.optimize_theta(x, y, theta)
    plots.plot_linear_regression_fit((theta, x, y))

    theta = np.array([[1.0], [1.0]])
    plots.plot_learning_curve((training_set, validation_set, theta))

    with open("result.txt", "w") as output_file:
        output_file.write("Part 1:\n")
        output_file.write("Cost for lambda=3:\n")
        output_file.write(f"{cost: .2f}\n")

        output_file.write("Gradient for lambda=1:\n")
        output_file.write(f"({gradient[0][0]: .2f}, {gradient[-1][0]: .2f})\n")


def part_2() -> None:
    """
    Polynomial regression:
    - Learning Polynomial Regression.
    - Adjusting the regularization parameter.
    - Selecting lambda using a cross validation set.
      
    Returns:
      None
    """
    training_set, validation_set, test_set = readers.read_data(DATA_PATH)
    x, y = training_set

    p = 5
    x_extended = algorithms.construct_polynomial_matrix(x, p)
    x_norm, means, stds = algorithms.normalize_features(x_extended)

    theta = np.ones((x_norm.shape[1], 1))

    for _lambda in (0, 1, 100):
        theta_optimized = algorithms.optimize_theta(x_norm, y, theta, _lambda=_lambda)
        plots.plot_polynomial_regression_fit(
            (theta_optimized, x_extended, y, means, stds, _lambda)
        )
        plots.plot_polynomial_learning_curve(
            (training_set, validation_set, theta), _lambda=_lambda
        )


def main() -> None:
    """
    The main functions. Calls the functions that implement different parts of the solution
    to the Task 5 from the coding homeworks from the Machine Learning course on coursera.com.

    Returns:
      None
    """
    plt.style.use("seaborn")
    part_1()
    part_2()
    plt.show()


if __name__ == "__main__":
    main()
