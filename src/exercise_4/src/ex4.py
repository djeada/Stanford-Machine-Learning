""""
The goal of this module is to give a comprehensive solution to Task 4
from the coding homeworks from the Machine Learning course on coursera.com.
The task is broken down into smaller parts.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import readers
import algorithms
import plots

DATA_PATH_1 = Path("../data/data.mat")
DATA_PATH_2 = Path("../data/weights.mat")


def part_1() -> None:
    """
    Neural network implementation:
    - Visualizing the data
    - Feedforward algorithm and cost function computation
    - Regularized cost function

    Returns:
      None
    """
    x, y = readers.read_data(DATA_PATH_1)
    theta = readers.read_weights(DATA_PATH_2)

    theta_1, theta_2 = theta
    theta = (theta_1, theta_2)
    plots.display_random_grid(x)
    readers.clean_y(y)
    a, z = algorithms.propagate_forward(x, theta)

    with open("../result.txt", "w") as output_file:
        output_file.write("Part 1:\n")
        output_file.write("Computed cost with theta loaded from weights.mat file:\n")
        output_file.write(f"{algorithms.compute_cost(a[-1], y, theta): .2f}\n")

        output_file.write("Computed cost with regularization parameter (lambda=1):\n")
        output_file.write(f"{algorithms.compute_cost(a[-1], y, theta, 1): .2f}\n")


def part_2() -> None:
    """
    Neural network enhanced with backpropagation algorithm:
    - Computation of sigmoid gradient
    - Gradient checking
    - Neural Network regularization

    Returns:
      None
    """
    x, y = readers.read_data(DATA_PATH_1)

    input_layer_size = 400
    hidden_layer_size = 25
    num_labels = 10

    theta_1 = algorithms.random_weights(input_layer_size, hidden_layer_size)
    theta_2 = algorithms.random_weights(hidden_layer_size, num_labels)
    theta = np.concatenate([theta_1.ravel(), theta_2.ravel()], axis=0)
    reshaped_theta = algorithms.split_theta(
        theta, input_layer_size, hidden_layer_size, num_labels
    )

    readers.clean_y(y)

    a, z = algorithms.propagate_forward(x, reshaped_theta)
    gradient = algorithms.backpropagation(a, z, y, reshaped_theta)

    theta = algorithms.optimize_theta(
        x, y, theta, input_layer_size, hidden_layer_size, num_labels
    )
    reshaped_theta = algorithms.split_theta(
        theta, input_layer_size, hidden_layer_size, num_labels
    )
    x = np.delete(x, 0, 1)
    accuracy = algorithms.calculate_accuracy(x, y, reshaped_theta) * 100

    with open("../result.txt", "a") as output_file:
        output_file.write("Part 2:\n")
        output_file.write("Backpropagation results:\n")

        output_file.write("Last 10 elements of delta 1:\n")
        for val in gradient[0][-10:]:
            output_file.write(f"{val: .5f}\n")

        output_file.write("Last 10 elements of delta 2:\n")
        for val in gradient[1][-10:]:
            output_file.write(f"{val: .5f}\n")

        output_file.write("Training set accuracy:\n")
        output_file.write(f"{accuracy: .2f}\n")


def main() -> None:
    """
    The main functions. Calls the functions that implement different parts of the solution
    to the Task 4 from the coding homeworks from the Machine Learning course on coursera.com.

    Returns:
      None
    """
    plt.style.use("seaborn")
    part_1()
    part_2()
    plt.show()


if __name__ == "__main__":
    main()
