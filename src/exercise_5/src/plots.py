""""
The goal of this module is to implement all the visualization
tools needed to graph the data and results of the computations
for the Task 5 from the coding homeworks in the Machine Learning
course on coursera.com.
"""

from typing import Tuple, Type
import numpy as np
import matplotlib.pyplot as plt
import copy

import algorithms


def plot_data(
    data: Tuple[np.ndarray, np.ndarray], title: str = "Scatter Plot Of Training Data"
) -> None:
    """
    Plots the data as a scatter plot.

    :param data: The data to plot.
    :param title: The title of the plot.
    """
    x, y = data
    plt.figure(figsize=(8, 6))
    plt.scatter(x[:, 1], y, c="red", s=50)
    plt.ylabel("Water flowing out of the dam (y)")
    plt.xlabel("Change in water level (x)")
    plt.title(title)
    plt.savefig(title.lower().replace(" ", "_"))


def plot_regression_fit(
    x: np.ndarray,
    y: np.ndarray,
    model: algorithms.LinearRegression,
    title: str = "Training Data With Linear Regression Fit",
    x_label="x",
    y_label="y",
    x_min: float = -50,
    x_max: float = 40,
) -> None:

    """
    Plots the training data and the regression fit.

    :param x: The x values of the data.
    :param y: The y values of the data.
    :param model: The model used to fit the data.
    :param title: The title of the plot.
    :param x_label: The label of the x axis.
    :param y_label: The label of the y axis.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(x[:, 1], y, marker="x", c="r", s=100)
    domain = np.linspace(x_min, x_max, 100)
    plt.plot(domain, model.predict(np.c_[np.ones(len(domain)), domain]), c="b")
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.xlim(x_min, x_max)
    plt.title(title)
    plt.legend(["Training Data", str(model)])
    plt.show()


def plot_linear_regression_learning_curve(
    x: np.ndarray,
    y: np.ndarray,
    xval: np.ndarray,
    yval: np.ndarray,
    orignal_model: algorithms.LinearRegression,
    title: str = "Learning Curve",
    x_label="Number of training examples",
    y_label="Error",
) -> None:

    """
    Plots the learning curve for a given model.

    :param x: The x values of the training data.
    :param y: The y values of the training data.
    :param xval: The x values of the cross validation data.
    :param yval: The y values of the cross validation data.
    :param orignal_model: The model used to fit the data.
    :param title: The title of the plot.
    :param x_label: The label of the x axis.
    :param y_label: The label of the y axis.
    """
    plt.figure(figsize=(8, 6))

    costs = []
    val_costs = []

    for i in range(1, 13):
        model = copy.deepcopy(orignal_model)
        model.fit(x[:i], y[:i])
        costs.append(model.compute_cost(x[:i], y[:i]))
        val_costs.append(model.compute_cost(xval, yval))

    plt.plot(range(1, 13), costs, c="b", label="Training error")
    plt.plot(range(1, 13), val_costs, c="r", label="Cross validation error")

    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.legend()
    plt.show()


def plot_polynomial_regression_learning_curve(
    x: np.ndarray,
    y: np.ndarray,
    xval: np.ndarray,
    yval: np.ndarray,
    orignal_model: algorithms.PolynomialRegression,
    title: str = "Learning Curve",
    x_label="Number of training examples",
    y_label="Error",
) -> None:

    """
    Plots the learning curve for a given model.

    :param x: The x values of the training data.
    :param y: The y values of the training data.
    :param xval: The x values of the cross validation data.
    :param yval: The y values of the cross validation data.
    :param orignal_model: The model used to fit the data.
    :param title: The title of the plot.
    :param x_label: The label of the x axis.
    :param y_label: The label of the y axis.
    """
    plt.figure(figsize=(8, 6))

    costs = []
    val_costs = []

    for i in range(1, 13):
        model = copy.deepcopy(orignal_model)
        model.fit(x[:i], y[:i])
        costs.append(model.compute_cost(model.process_input(x[:i]), y[:i]))
        val_costs.append(model.compute_cost(model.process_input(xval), yval))

    plt.plot(range(1, 13), costs, c="b", label="Training error")
    plt.plot(range(1, 13), val_costs, c="r", label="Cross validation error")

    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.ylim(0, 150)
    plt.title(title)
    plt.legend()
    plt.show()


def plot_polynomial_regression_learning_curve_for_lambdas(
    x: np.ndarray,
    y: np.ndarray,
    xval: np.ndarray,
    yval: np.ndarray,
    lambdas: Tuple[float, ...],
    title: str = "Learning Curve",
    x_label="Number of training examples",
    y_label="Error",
) -> None:

    """
    Plots the learning curve for a given model.

    :param x: The x values of the training data.
    :param y: The y values of the training data.
    :param xval: The x values of the cross validation data.
    :param yval: The y values of the cross validation data.
    :param orignal_model: The model used to fit the data.
    :param lambdas: The lambdas to use.
    :param title: The title of the plot.
    :param x_label: The label of the x axis.
    :param y_label: The label of the y axis.
    """
    plt.figure(figsize=(8, 6))

    costs = []
    val_costs = []

    for _lambda in lambdas:
        model = algorithms.PolynomialRegression(5, _lambda=_lambda)
        model.fit(x, y)
        costs.append(model.compute_cost(model.process_input(x), y))
        val_costs.append(model.compute_cost(model.process_input(xval), yval))

    plt.plot(lambdas, costs, c="b", label="Training error")
    plt.plot(lambdas, val_costs, c="r", label="Cross validation error")

    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.legend()
    plt.show()
