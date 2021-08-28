import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.optimize

DATA_PATH = "data/data.mat"


def read_data(path, usecols=(0, 1)):
    """
    x is a matrix with m rows and n columns
    y is a matrix with m rows and 1 column
    three input sets: training, validation and test 
    """

    raw_data = scipy.io.loadmat(path)
    x, y = raw_data["X"], raw_data["y"]
    x_validation, y_validation = raw_data["Xval"], raw_data["yval"]
    x_test, y_test = raw_data["Xtest"], raw_data["ytest"]

    x = np.insert(x, 0, np.ones(len(x)), axis=1)
    x_validation = np.insert(x_validation, 0, np.ones(len(x_validation)), axis=1)
    x_test = np.insert(x_test, 0, np.ones(len(x_test)), axis=1)

    return (x, y), (x_validation, y_validation), (x_test, y_test)


def hypothesis_function(x, theta):
    return np.dot(x, theta)


def compute_cost(x, y, theta=None, _lambda=0):
    if theta is None:
        theta = np.zeros((x.shape[1], 1))

    m = y.size

    j = (
        1
        / (2 * m)
        * np.dot(
            (hypothesis_function(x, theta).reshape((m, 1)) - y).T,
            (hypothesis_function(x, theta).reshape((m, 1)) - y),
        )
    )
    reg = _lambda / (2 * m) * np.dot(theta[1:].T, theta[1:])

    return (j + reg)[0][0]


def compute_gradient(x, y, theta=None, _lambda=0):
    if theta is None:
        theta = np.zeros((x.shape[1], 1))

    m = y.size
    theta = theta.reshape((theta.shape[0], 1))

    gradient = 1 / m * np.dot(x.T, hypothesis_function(x, theta) - y)
    reg = _lambda / m * theta

    # don't regularize bias term
    reg[0] = 0

    return gradient + reg.reshape((gradient.shape[0], 1))


def optimize_theta(x, y, theta, _lambda=0):

    return scipy.optimize.minimize(
        lambda _theta: compute_cost(x, y, _theta, _lambda),
        x0=theta,
        method="BFGS",
        jac=None,
        tol=None,
        callback=None,
        options={
            "disp": False,
            "gtol": 1e-05,
            "eps": 1.4901161193847656e-08,
            "return_all": False,
            "maxiter": None,
        },
    ).x


def construct_polynomial_matrix(x, p):
    """
    Takes an x matrix and returns an x matrix with additional columns.
    First additional column is 2'nd column with all values squared,
    the next additional column is 2'nd column with all values cubed etc. 
    """

    p_matrix = x.copy()

    for i in range(2, p + 2):
        p_matrix = np.insert(p_matrix, p_matrix.shape[1], np.power(x[:, 1], i), axis=1)

    return p_matrix


def normalize_features(x):

    x_norm = x.copy()

    feature_means = np.mean(x_norm, axis=0)
    x_norm[:, 1:] = x_norm[:, 1:] - feature_means[1:]
    feature_stds = np.std(x_norm, axis=0, ddof=1)
    x_norm[:, 1:] = x_norm[:, 1:] / feature_stds[1:]

    return x_norm, feature_means, feature_stds


def plot_data(data):
    x, y = data
    plt.figure(figsize=(10, 6))
    plt.scatter(x[:, 1], y, c="red", s=50)
    plt.ylabel("Water flowing out of the dam (y)")
    plt.xlabel("Change in water level (x)")
    plt.savefig("scatter_plot_of_training_data")


def plot_linear_regression_fit(data):
    def fit(theta, alpha):
        return theta[0] + theta[1] * alpha

    theta, x, y = data

    plt.figure(figsize=(10, 6))

    plt.scatter(x[:, 1], y, c="red", s=50)
    plt.plot(x[:, 1], fit(theta, x[:, 1]), "b-", label="Linear regression")
    plt.ylabel("Water flowing out of the dam (y)")
    plt.xlabel("Change in water level (x)")
    plt.legend(loc="lower right")
    plt.savefig("linear_regression_fit")


def plot_learning_curve(data):

    training_set, validation_set, theta = data
    x, y = training_set
    x_validation, y_validation = validation_set

    m_history = list()
    train_error_history = list()
    validation_error_history = list()

    for i in range(1, 12):
        x_subset = x[:i, :]
        y_subset = y[:i]
        theta_optimized = optimize_theta(x_subset, y_subset, theta)

        m_history.append(y_subset.shape[0])
        train_error_history.append(compute_cost(x_subset, y_subset, theta_optimized))
        validation_error_history.append(
            compute_cost(x_validation, y_validation, theta_optimized)
        )

    plt.figure(figsize=(10, 6))
    plt.plot(m_history, train_error_history, label="Train")
    plt.plot(m_history, validation_error_history, label="Cross Validation")
    plt.xlabel("Number of training examples")
    plt.ylabel("Error")
    plt.title("Learning curve for linear regression")
    plt.legend()
    plt.savefig("learning_curve")


def plot_polynomial_regression_fit(data, n_points=50):

    theta, x, y, means, stds, _lambda = data

    x_range = np.linspace(-50, 50, n_points)

    x_temp = np.ones((len(x_range), 1))
    x_temp = np.insert(x_temp, x_temp.shape[1], x_range.T, axis=1)
    x_temp = construct_polynomial_matrix(x_temp, len(theta) - 2)

    x_temp[:, 1:] -= means[1:]
    x_temp[:, 1:] /= stds[1:]

    plt.figure(figsize=(10, 6))
    plt.scatter(x[:, 1], y, c="red", s=50)
    plt.plot(
        x_range,
        hypothesis_function(x_temp, theta),
        "b--",
        label="Polynomial regression",
    )
    plt.ylabel("Water flowing out of the dam (y)")
    plt.xlabel("Change in water level (x)")
    plt.legend(loc="lower right")
    plt.title(f"polynomial regression fit (lambda={_lambda})")
    plt.savefig(f"polynomial_regression_fit_{_lambda}")


def plot_polynomial_learning_curve(data, _lambda=0, p=5):

    training_set, validation_set, theta = data
    x, y = training_set
    x_validation, y_validation = validation_set
    x_validation, _, __ = normalize_features(
        construct_polynomial_matrix(x_validation, p)
    )

    m_history = list()
    train_error_history = list()
    validation_error_history = list()

    for i in range(1, 12):
        x_subset = x[:i, :]
        x_subset = construct_polynomial_matrix(x_subset, p)
        x_subset, _, __ = normalize_features(x_subset)
        y_subset = y[:i]
        theta_optimized = optimize_theta(x_subset, y_subset, theta, _lambda)

        m_history.append(y_subset.shape[0])
        train_error_history.append(
            compute_cost(x_subset, y_subset, theta_optimized, _lambda)
        )
        validation_error_history.append(
            compute_cost(x_validation, y_validation, theta_optimized, _lambda)
        )

    plt.figure(figsize=(10, 6))
    plt.plot(m_history, train_error_history, label="Train")
    plt.plot(m_history, validation_error_history, label="Cross Validation")
    plt.xlabel("Number of training examples")
    plt.ylabel("Error")
    plt.title(f"Learning curve for polynomial regression (lambda={_lambda})")
    plt.legend()
    plt.savefig(f"polynomial_learning_curve_lambda_{_lambda}")


def part_1():
    training_set, validation_set, test_set = read_data(DATA_PATH)
    plot_data(training_set)

    x, y = training_set
    theta = np.array([[1.0], [1.0]])
    cost = compute_cost(x, y, theta, 3)
    gradient = compute_gradient(x, y, theta, 1)

    theta = optimize_theta(x, y, theta)
    plot_linear_regression_fit((theta, x, y))

    theta = np.array([[1.0], [1.0]])
    plot_learning_curve((training_set, validation_set, theta))

    with open("result.txt", "w") as output_file:
        output_file.write("Part 1:\n")
        output_file.write("Cost for lambda=3:\n")
        output_file.write(f"{cost: .2f}\n")

        output_file.write("Gradient for lambda=1:\n")
        output_file.write(f"({gradient[0][0]: .2f}, {gradient[-1][0]: .2f})\n")


def part_2():
    training_set, validation_set, test_set = read_data(DATA_PATH)
    x, y = training_set

    p = 5
    x_extended = construct_polynomial_matrix(x, p)
    x_norm, means, stds = normalize_features(x_extended)

    theta = np.ones((x_norm.shape[1], 1))

    for _lambda in (0, 1, 100):
        theta_optimized = optimize_theta(x_norm, y, theta, _lambda=_lambda)
        plot_polynomial_regression_fit(
            (theta_optimized, x_extended, y, means, stds, _lambda)
        )
        plot_polynomial_learning_curve(
            (training_set, validation_set, theta), _lambda=_lambda
        )


def main():
    plt.style.use("seaborn")
    part_1()
    part_2()
    # plt.show()


if __name__ == "__main__":
    main()
