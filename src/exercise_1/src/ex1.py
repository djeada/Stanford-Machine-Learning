""""
The goal of this module is to give a comprehensive solution to Task 1
from the coding homeworks in from Machine Learning course on coursera.com.
The task is broken down into smaller parts.
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import exercise_1.src.preprocessing as preprocessing
import algorithms
import plots

DATA_PATH_1 = Path("../data/data1.txt")
DATA_PATH_2 = Path("../data/data2.txt")


def part_1() -> None:
    """
    Linear regression with one variable.
    - Plotting the Data.
    - Calculating cost using gradient descent algorithm.
    - Plotting the convergence of the cost function.
    - Plotting the linear regression fit.
    - Visualizing the cost with respect to theta.

    Returns:
      None
    """

    x, y = preprocessing.read_data(DATA_PATH_1)
    # plots.plot_data((x, y))
    theta_history, costs = algorithms.gradient_descent(x, y)
    # plots.plot_convergence(costs, (4, 7), "Convergence Of Gradient Descent Part1")
    # plots.plot_linear_regression_fit((theta_history[-1], x, y))
    # plots.plot_cost_function_3d((theta_history, costs, x, y))

    with open("../result.txt", "w") as output_file:
        # the expected value of cost is 32.07
        cost = algorithms.compute_cost(x, y)
        output_file.write("Part 1:\n")
        output_file.write("Computed cost with initial value of theta:\n")
        output_file.write(f"{cost: .2f}\n")

        # the expected value of cost is 17.91
        cost = algorithms.compute_cost(x, y, theta_history[-1])
        output_file.write("Computed cost with optimized value of theta:\n")
        output_file.write(f"{cost: .2f}\n")


def part_2() -> None:
    """
    Linear regression with multiple variables.
    - Feature normalization and visualisation trough a histogram.
    - Calculating cost using gradient descent algorithm.
    - Plotting the convergence of the cost function.
    - Making predictions using normal equations.

    Returns:
      None
    """

    x, y = preprocessing.read_data(DATA_PATH_2, (0, 1, 2))
    plots.plot_histogram(x, "Count Freq Per Column Before Norm")

    x_norm = x.copy()
    means, stds = algorithms.normalize_features(x_norm)
    plots.plot_histogram(x_norm, "Count Freq Per Column After Norm")
    theta_history, costs = algorithms.gradient_descent(x_norm, y)
    theta = np.array(theta_history[-1])

    house_size = 1650
    num_bedrooms = 3

    plots.plot_convergence(
        costs, (0, max(costs)), "Convergence Of Gradient Descent Part2"
    )

    with open("../result.txt", "a") as output_file:
        # the expected result is 293083
        result = algorithms.predict_using_normalized_features(
            means, stds, theta, house_size, num_bedrooms
        )
        output_file.write("\nPart 2:\n")
        output_file.write(
            "Prediction using means and stds for price of house with 1650 square feet and 3 bedrooms:\n"
        )
        output_file.write(f"{result: .2f}$\n")

        # the expected result is 293083
        result = algorithms.predict_from_normal_equation(
            x, y, house_size, num_bedrooms
        )[0]
        output_file.write(
            "\nNormal equation prediction for price of house with 1650 square feet and 3 bedrooms:\n"
        )
        output_file.write(f"{result: .2f}$\n")


def main() -> None:
    """
    The main functions. Calls the functions that implement different parts of the solution
    to the Task 1 from the coding homeworks from the Machine Learning course on coursera.com.

    Returns:
      None
    """
    plt.style.use("seaborn")
    part_1()
    part_2()
    plt.show()


if __name__ == "__main__":
    main()
