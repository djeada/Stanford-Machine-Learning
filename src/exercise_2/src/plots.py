"""
The goal of this module is to implement all the visualization
tools needed to graph the data and results of the computations
for the Task 2 from the coding homeworks in the Machine Learning
course on coursera.com.
"""
import numpy as np
import matplotlib.pyplot as plt
import algorithms
from preprocessing import FeatureMapper


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
        positive[:, 1], positive[:, 2], marker="+", c="k", s=150, label=label_positive
    )
    plt.scatter(
        negative[:, 1], negative[:, 2], marker="o", c="y", s=100, label=label_negative
    )
    plt.ylabel(label_y)
    plt.xlabel(label_x)
    plt.legend(loc="upper right")
    plt.title(title)
    plt.savefig(title.lower().replace(" ", "_"))


def plot_cost_function(
    cost_history: np.ndarray, title: str = "Cost Function Convergence"
) -> None:
    """
    Plot the cost function convergence.

    :param cost_history: The cost function history.
    :param title: The title of the plot.
    """

    plt.figure(figsize=(10, 6))
    plt.scatter(np.arange(cost_history.size), cost_history, marker="o", c="g", s=10)
    plt.xlabel("Iterations")
    plt.ylabel("Cost J")
    plt.title(title)
    plt.show()


def plot_logistic_regression_fit(
    positive: np.ndarray,
    negative: np.ndarray,
    model: algorithms.LogisticRegression,
    title: str = "Training Data With Logistic Regression Fit",
    label_x: str = "x_1",
    label_y: str = "x_2",
    label_positive: str = "positive",
    label_negative: str = "negative",
) -> None:
    """
    Plot the training data and the logistic regression fit.

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
        positive[:, 1], positive[:, 2], marker="+", c="k", s=150, label=label_positive
    )
    plt.scatter(
        negative[:, 1], negative[:, 2], marker="o", c="y", s=100, label=label_negative
    )
    plt.ylabel(label_y)
    plt.xlabel(label_x)
    plt.legend(loc="upper right")
    plt.title(title)
    domain = np.array([np.min(negative[:, 1]), np.max(positive[:, 1])])
    plt.plot(domain, model.boundary(domain), c="b")
    plt.legend(["positive", "negative", f"{model}"])
    plt.show()


def plot_boundary_as_contour(
    positive: np.ndarray,
    negative: np.ndarray,
    model: algorithms.LogisticRegression,
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
        positive[:, 1], positive[:, 2], marker="+", c="k", s=150, label=label_positive
    )
    plt.scatter(
        negative[:, 1], negative[:, 2], marker="o", c="y", s=100, label=label_negative
    )
    plt.ylabel(label_y)
    plt.xlabel(label_x)
    plt.legend(loc="upper right")
    plt.title(title)

    # Create grid coordinates for plotting
    u = np.linspace(np.min(positive[:, 1]), np.max(positive[:, 1]), 50)
    v = np.linspace(np.min(negative[:, 2]), np.max(negative[:, 2]), 50)
    z = np.zeros((u.size, v.size))
    # Evaluate z = theta*x over the grid
    mapper = FeatureMapper()
    for i, ui in enumerate(u):
        for j, vj in enumerate(v):
            # z[i, j] = model.predict(algorithms.map_feature(np.array([ui]), np.array([vj])))
            z[i, j] = mapper.map(np.array([ui]), np.array([vj])) @ model.theta

    z = z.T  # important to transpose z before calling contour

    # Plot z = 0
    # Notice you need to specify the range [0, 0]
    plt.contour(u, v, z, levels=[0], linewidths=2, colors="g")
    plt.show()
