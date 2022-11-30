""""
The goal of this module is to implement all the visualization
tools needed to graph the data and results of the computations
for the Task 6 from the coding homeworks in the Machine Learning
course on coursera.com.
"""

import numpy as np
import matplotlib.pyplot as plt
import algorithms


def plot_data(
    positive: np.ndarray,
    negative: np.ndarray,
    title: str = "Training Data",
    label_x: str = "x_1",
    label_y: str = "x_2",
    label_positive: str = "positive",
    label_negative: str = "negative",
) -> None:
    """
    Plot the training data.
    :param positive: The training data.
    :param negative: The training labels.
    :param title: The title of the plot.
    :param label_x: The label of the x-axis.
    :param label_y: The label of the y-axis.
    :param label_positive: The label of the positive data.
    :param label_negative: The label of the negative data.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(
        positive[:, 0], positive[:, 1], marker="+", c="k", s=150, label=label_positive
    )
    plt.scatter(
        negative[:, 0], negative[:, 1], marker="o", c="y", s=100, label=label_negative
    )
    plt.ylabel(label_y)
    plt.xlabel(label_x)
    plt.legend(loc="upper right")
    plt.title(title)
    plt.show()


def plot_data_and_svm_boundary(
    positive: np.ndarray,
    negative: np.ndarray,
    model: algorithms.SvmRegressionBase,
    title: str = "Decision Boundary",
    label_x: str = "x_1",
    label_y: str = "x_2",
    label_positive: str = "positive",
    label_negative: str = "negative",
) -> None:
    """
    Plots the training data and the decision boundary of the SVM regression model.

    :param positive: The training data.
    :param negative: The training labels.
    :param model: The SVM regression model.
    :param title: The title of the plot.
    :param label_x: The label of the x-axis.
    :param label_y: The label of the y-axis.
    :param label_positive: The label of the positive data.
    :param label_negative: The label of the negative data.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(
        positive[:, 0], positive[:, 1], marker="+", c="k", s=150, label=label_positive
    )
    plt.scatter(
        negative[:, 0], negative[:, 1], marker="o", c="y", s=100, label=label_negative
    )

    x = np.linspace(plt.xlim()[0], plt.xlim()[1], 100)
    y = model.boundary(x)
    plt.plot(x, y, label=str(model))
    plt.ylabel(label_y)
    plt.xlabel(label_x)
    plt.legend()
    plt.title(title)
    plt.show()


def plot_boundary_as_contour(
    positive: np.ndarray,
    negative: np.ndarray,
    model: algorithms.SvmRegressionBase,
    title: str = "Decision Boundary",
    label_x: str = "x_1",
    label_y: str = "x_2",
    label_positive: str = "positive",
    label_negative: str = "negative",
) -> None:
    """
    Plot the decision boundary as a contour plot.
    :param positive: The training data.
    :param negative: The training labels.
    :param model: The trained logistic regression model.
    :param title: The title of the plot.
    :param label_x: The label of the x-axis.
    :param label_y: The label of the y-axis.
    :param label_positive: The label of the positive data.
    :param label_negative: The label of the negative data.
    """

    plt.figure(figsize=(10, 6))
    plt.scatter(
        positive[:, 0], positive[:, 1], marker="+", c="k", s=150, label=label_positive
    )
    plt.scatter(
        negative[:, 0], negative[:, 1], marker="o", c="y", s=100, label=label_negative
    )

    plt.ylabel(label_y)
    plt.xlabel(label_x)
    plt.legend(loc="upper right")
    plt.title(title)

    x_min, x_max = plt.xlim()
    x_min, x_max = x_min - 0.1, x_max + 0.1
    x1 = np.linspace(x_min, x_max, 100)
    x2 = np.linspace(x_min, x_max, 100)
    x1, x2 = np.meshgrid(x1, x2)
    z = model.boundary(np.c_[x1.ravel(), x2.ravel()])
    z = z.reshape(x1.shape)
    plt.contour(x1, x2, z, [0], linewidths=2, colors="g")
    plt.show()
