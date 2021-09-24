""""
The goal of this module is to give a comprehensive solution to Task 1
from the coding homeworks in the Machine Learning course on coursera.com.
The task is broken down into smaller parts.
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import readers
import algorithms
import plots

DATA_PATH_1 = Path("../data/data1.txt")
DATA_PATH_2 = Path("../data/data2.txt")


def part_1() -> None:
    """
    """

    x, y = readers.read_data(DATA_PATH_1)
    plots.plot_data((x, y))
    cost = algorithms.compute_cost(x, y)
    theta_history, costs = algorithms.gradient_descent(x, y)
    plots.plot_convergence(costs, (4, 7), Path("convergence_of_gradient_descent_part1"))
    plots.plot_linear_regression_fit((theta_history[-1], x, y))
    plots.plot_cost_function_3d((theta_history, costs, x, y))

    with open("../result.txt", "w") as output_file:
        output_file.write("Part 1:\n")
        output_file.write("Computed cost:\n")
        output_file.write(f"{cost: .2f}\n")


def part_2() -> None:
    """
    """

    x, y = readers.read_data(DATA_PATH_2, (0, 1, 2))
    plots.plot_histogram(x, Path("count_freq_per_column_before_norm"))

    x_norm = x.copy()
    means, stds = algorithms.normalize_features(x_norm)
    plots.plot_histogram(x, Path("count_freq_per_column_after_norm"))
    theta_history, costs = algorithms.gradient_descent(x_norm, y)
    theta = np.array(theta_history[-1])

    house_size = 1650
    num_bedrooms = 3

    plots.plot_convergence(costs, (0, max(costs)), Path("convergence_of_gradient_descent_part2"))

    result = algorithms.predict_from_means_and_stds(
        means, stds, theta, house_size, num_bedrooms
    )

    with open("../result.txt", "a") as output_file:
        output_file.write("\nPart 2:\n")
        output_file.write(
            "Prediction using means and stds for price of house with 1650 square feet and 3 bedrooms:\n"
        )
        output_file.write(f"{result: .2f}$\n")

        result = algorithms.predict_from_normal_equation(x, y, house_size, num_bedrooms)[0]
        output_file.write(
            "\nNormal equation prediction for price of house with 1650 square feet and 3 bedrooms:\n"
        )
        output_file.write(f"{result: .2f}$\n")


def main() -> None:
    """
    """
    plt.style.use("seaborn")
    part_1()
    part_2()
    plt.show()


if __name__ == "__main__":
    main()
