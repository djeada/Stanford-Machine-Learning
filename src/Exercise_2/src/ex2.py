""""
The goal of this module is to give a comprehensive solution to Task 2
from the coding homeworks in the Machine Learning course on coursera.com.
The task is broken down into smaller parts.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import readers
import algorithms
import plots

DATA_PATH_1 = Path("../data/data1.txt")
DATA_PATH_2 = Path("../data/data2.txt")


def part_1() -> None:
    x, y = readers.read_data(DATA_PATH_1)
    pos = np.array([x[i] for i in range(len(x)) if y[i] == 1])
    neg = np.array([x[i] for i in range(len(x)) if y[i] == 0])
    plots.plot_data((pos, neg), ("Admitted", "Not admitted", "Exam 1 score", "Exam 2 score"))
    cost = algorithms.compute_cost(x, y)

    theta = algorithms.optimize_theta(x, y)
    plots.plot_logistic_regression_fit((theta, x, y, pos, neg))

    with open("result.txt", "w") as output_file:
        output_file.write("Part 1:\n")
        output_file.write("Computed cost with the optimal parameters of theta:\n")
        output_file.write(f"{algorithms.compute_cost(x, y, theta): .2f}\n")

        output_file.write(
            "\nFor a student with an Exam 1 score of 45 and an Exam 2 score of 85, computed admission probability is:\n"
        )
        output_file.write(f"{algorithms.predict(theta, 45, 85): .2f}\n")

        output_file.write("\nCorrect samples in training data:\n")
        output_file.write(f"{algorithms.correct_predictions(theta, pos, neg): .2f}\n")


def part_2() -> None:
    x, y = readers.read_data(DATA_PATH_2)
    pos = np.array([x[i] for i in range(len(x)) if y[i] == 1])
    neg = np.array([x[i] for i in range(len(x)) if y[i] == 0])
    plots.plot_data((pos, neg), ("y=1", "y=0", "Microchip Test 1", "Microchip Test 2"))
    x_mapped = algorithms.map_feature(x[:, 1], x[:, 2])

    for _lambda in (1, 10, 100):
        theta = algorithms.optimize_regularized_theta(x_mapped, y, _lambda=_lambda)
        plots.plot_data(
            (pos, neg),
            ("y=1", "y=0", "Microchip Test 1", "Microchip Test 2"),
            f"decision_boundary_lambda_{_lambda}",
        )
        plots.plot_boundary((theta, _lambda))

    with open("result.txt", "a") as output_file:
        output_file.write("\nPart 2:\n")
        output_file.write("Computed cost with mapped X:\n")
        output_file.write(f"{algorithms.compute_cost(x_mapped, y): .2f}\n")


def main() -> None:
    plt.style.use("seaborn")
    part_1()
    part_2()
    plt.show()


if __name__ == "__main__":
    main()
