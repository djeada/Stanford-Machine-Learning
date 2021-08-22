import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm
import itertools

DATA_PATH_1 = "data/data1.txt"
DATA_PATH_2 = "data/data2.txt"


def read_data(path, usecols=(0, 1)):
    """
    x is a matrix with 2 columns and m rows
    y is a matrix with m rows and 1 column
    """

    raw_data = np.loadtxt(path, delimiter=",", usecols=usecols, unpack=True)
    x = np.transpose(np.array(raw_data[:-1]))
    ones = np.ones(len(x))
    # prepend column of ones
    x = np.insert(x, 0, ones, axis=1)
    y = np.transpose(np.array(raw_data[-1:]))

    return x, y


def hypothesis_function(x, theta):
    return np.dot(x, theta)


def compute_cost(x, y, theta=None):
    if theta is None:
        theta = np.zeros((x.shape[1], 1))

    return (
        1.0
        / (2 * y.size)
        * np.dot(
            (hypothesis_function(x, theta) - y).T, (hypothesis_function(x, theta) - y)
        )
    )[0][0]


def gradient_descent(x, y, theta=None, num_iter=2000, alpha=0.01):
    if theta is None:
        theta = np.zeros((x.shape[1], 1))

    initial_theta = theta
    m = y.size
    costs = []
    theta_history = []

    for _ in range(num_iter):
        costs.append(compute_cost(x, y, theta))
        theta_history.append(list(theta[:, 0]))

        for i in range(len(theta)):
            theta[i] -= (alpha / m) * np.sum(
                (hypothesis_function(x, initial_theta) - y)
                * np.array(x[:, i]).reshape(m, 1)
            )

    return theta_history, costs


def normalize_features(x):
    means = [np.mean(x[:, 0])]
    stds = [np.std(x[:, 0])]

    for i in range(1, x.shape[1]):
        means.append(np.mean(x[:, i]))
        stds.append(np.std(x[:, i]))
        x[:, i] = (x[:, i] - means[-1]) / stds[-1]

    return means, stds


def predict_from_means_and_stds(means, stds, theta, house_size, num_bedrooms):
    y = np.array([house_size, num_bedrooms])
    y = [(y[i] - means[i + 1]) / stds[i + 1] for i in range(len(y))]
    y.insert(0, 1)

    return hypothesis_function(y, theta)


def predict_from_normal_equation(x, y, house_size, num_bedrooms):
    def norm_eq(x, y):
        return np.dot(np.dot(np.linalg.inv(np.dot(x.T, x)), x.T), y)

    return hypothesis_function([1, house_size, num_bedrooms], norm_eq(x, y))


def plot_data(data):
    x, y = data
    plt.figure(figsize=(10, 6))
    plt.plot(x[:, 1], y[:, 0], "rx", markersize=10)
    plt.ylim(-5, 25)
    plt.xlim(4, 24)
    plt.ylabel("Profit in $10,000s")
    plt.xlabel("Population of City in 10,000s")
    plt.savefig("scatter_plot_of_training_data")


def plot_convergence(data, ylim, output_file):
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(data)), data, c="green", s=1)
    plt.title("Convergence of Cost Function")
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost J")
    plt.ylim(ylim)
    plt.savefig(output_file)


def plot_linear_regression_fit(data):
    def fit(theta, alpha):
        return theta[0] + theta[1] * alpha

    theta, x, y = data

    plt.figure(figsize=(10, 6))
    plt.plot(x[:, 1], y[:, 0], "rx", markersize=10, label="Training data")
    plt.plot(x[:, 1], fit(theta, x[:, 1]), "b-", label="Linear regression")
    plt.ylabel("Profit in $10,000s")
    plt.xlabel("Population of City in 10,000s")
    plt.ylim(-5, 25)
    plt.xlim(4, 24)
    plt.legend(loc="lower right")
    plt.savefig("linear_regression_fit")


def plot_cost_function_3d(data):

    theta_history, costs, x, y = data

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection="3d")

    x_range = np.arange(-10, 10, 0.5)
    y_range = np.arange(-1, 4, 0.2)

    x_values = [_x for _x in x_range for _y in y_range]
    y_values = [_y for _x in x_range for _y in y_range]
    z_values = [
        compute_cost(x, y, np.array([[_x], [_y]])) for _x in x_range for _y in y_range
    ]

    ax.scatter(
        x_values, y_values, z_values, c=np.abs(z_values), cmap=plt.get_cmap("YlOrRd")
    )

    plt.plot(
        [theta[0] for theta in theta_history], [x[1] for x in theta_history], costs, "-"
    )

    plt.xlabel(r"$\theta_0$")
    plt.ylabel(r"$\theta_1$")
    plt.title("Cost Function")
    plt.savefig("surface_plot_of_cost_function")


def plot_histogram(x, output_file):

    plt.figure(figsize=(10, 6))

    for i in range(x.shape[-1]):
        plt.hist(x[:, i], label=f"col{i}")

    plt.xlabel("Column Value")
    plt.ylabel("Counts")
    plt.legend()
    plt.savefig(output_file)


def part_1():
    x, y = read_data(DATA_PATH_1)
    plot_data((x, y))
    # cost = compute_cost(x, y)
    theta_history, costs = gradient_descent(x, y)
    plot_convergence(costs, (4, 7), "convergence_of_gradient_descent_part1")
    plot_linear_regression_fit((theta_history[-1], x, y))
    plot_cost_function_3d((theta_history, costs, x, y))


def part_2():
    x, y = read_data(DATA_PATH_2, (0, 1, 2))
    plot_histogram(x, "count_freq_per_coulmn_before_norm")

    x_norm = x.copy()
    means, stds = normalize_features(x_norm)
    plot_histogram(x, "count_freq_per_coulmn_after_norm")
    theta_history, costs = gradient_descent(x_norm, y)

    house_size = 1650
    num_bedrooms = 3

    plot_convergence(costs, (0, max(costs)), "convergence_of_gradient_descent_part2")

    with open("result.txt", "a") as output_file:
        result = predict_from_means_and_stds(
            means, stds, theta_history[-1], house_size, num_bedrooms
        )
        output_file.write("Prediction using means and stds for price of house with 1650 square feet and 3 bedrooms:\n")
        output_file.write(f"{result: .2f}$\n")

        result = predict_from_normal_equation(x, y, house_size, num_bedrooms)[0]
        output_file.write("\nNormal equation prediction for price of house with 1650 square feet and 3 bedrooms:\n")
        output_file.write(f"{result: .2f}$\n")


def main():
    part_1()
    part_2()
    # plt.show()


if __name__ == "__main__":
    main()
