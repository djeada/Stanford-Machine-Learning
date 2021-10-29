""""
The goal of this module is to give a comprehensive solution to Task 3
from the coding homeworks from the Machine Learning course on coursera.com.
The task is broken down into smaller parts.
"""

import matplotlib.pyplot as plt
from pathlib import Path

import readers
import algorithms
import plots

DATA_PATH_1 = Path("../data/data.mat")
DATA_PATH_2 = Path("../data/weights.mat")


def part_1() -> None:
    """
    Logistic regression:
    - Classification with many classes.
    - Data visualization.
    - Vectorization of regularized logistic regression function.
    - Prediction accuracy in one-vs.-all situations.

    Returns:
      None
    """
    x, y = readers.read_data(DATA_PATH_1)
    plots.display_random_grid(x)
    theta = algorithms.train_model(x, y)
    accuracy = (
        algorithms.calculate_accuracy(x, y, theta, algorithms.predict_one_vs_all) * 100
    )

    with open("../result.txt", "w") as output_file:
        output_file.write("Part 1:\n")
        output_file.write("Training set accuracy:\n")
        output_file.write(f"{accuracy: .2f}\n")


def part_2() -> None:
    """
    Neural Networks:
    - Feedforward propagation and prediction.
    - Accuracy of a simple neural network.

    Returns:
      None
    """
    x, y = readers.read_data(DATA_PATH_1)
    theta = readers.read_weights(DATA_PATH_2)

    accuracy = algorithms.calculate_accuracy(x, y, theta, algorithms.predict_nn) * 100

    with open("../result.txt", "a") as output_file:
        output_file.write("Part 2:\n")
        output_file.write("Training set accuracy:\n")
        output_file.write(f"{accuracy: .2f}\n")


def main() -> None:
    """
    The main functions. Calls the functions that implement different parts of the solution
    to the Task 3 from the coding homeworks from the Machine Learning course on coursera.com.

    Returns:
      None
    """
    plt.style.use("seaborn")
    part_1()
    part_2()
    plt.show()


if __name__ == "__main__":
    main()
