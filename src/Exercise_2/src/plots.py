""""
The goal of this module is to implement all the visualization
tools needed to graph the data and results of the computations
for the Task 2 from the coding homeworks in the Machine Learning
course on coursera.com.
"""

import numpy as np
import matplotlib.pyplot as plt

import algorithms


def plot_data(data: tuple, labels, title: str = "scatter plot of training data") -> None:
    x, y = data
    label_pos, label_neg, label_x, label_y = labels
    plt.figure(figsize=(10, 6))
    plt.scatter(x[:, 1], x[:, 2], marker="+", c="k", s=150, label=label_pos)
    plt.scatter(y[:, 1], y[:, 2], marker="o", c="y", s=100, label=label_neg)
    plt.ylabel(label_y)
    plt.xlabel(label_x)
    plt.legend(loc="upper right")
    plt.title(title)
    plt.savefig(title.lower().replace(" ", "_"))


def plot_logistic_regression_fit(data: tuple, title: str = "logistic regression fit") -> None:
    def fit(_theta: np.ndarray, _x_range: np.ndarray) -> np.ndarray:
        return (-1.0 / _theta[2]) * (_theta[0] + _theta[1] * _x_range)

    theta, x, y, pos, neg = data
    x_range = np.array([np.min(x[:, 1]), np.max(x[:, 1])])

    plt.figure(figsize=(10, 6))
    plt.scatter(pos[:, 1], pos[:, 2], marker="+", c="k", s=150, label="Admitted")
    plt.scatter(neg[:, 1], neg[:, 2], marker="o", c="y", s=100, label="Not admitted")
    plt.plot(x_range, fit(theta, x_range), "b-")
    plt.ylabel("Exam 2 score")
    plt.xlabel("Exam 1 score")
    plt.legend(loc="upper right")
    plt.title(title)
    plt.savefig(title.lower().replace(" ", "_"))


def plot_boundary(data: tuple, title: str = "Decision Boundary") -> None:
    theta, _lambda = data

    range_x = np.linspace(-1, 1.5)
    range_y = np.linspace(-1, 1.5)
    range_z = [
        [
            np.dot(theta, algorithms.map_feature(np.array([y]), np.array([x])).T)[0][0]
            for x in range_x
        ]
        for y in range_y
    ]
    range_z = np.array(range_z).T

    contour = plt.contour(
        range_x, range_y, range_z, [0], cmap=plt.cm.coolwarm, extend="both"
    )
    plt.clabel(contour, inline=1, fmt={0: f"Lambda = {_lambda}"})
    plt.title(title)
    plt.savefig(title.lower().replace(" ", "_"))
