import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

DATA_PATH_1 = "data/data1.txt"
DATA_PATH_2 = "data/data2.txt"


def read_data(path, usecols=(0, 1, 2)):
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
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    return sigmoid(np.dot(x, theta))


def compute_cost(x, y, theta=None, _lambda=0):
    if theta is None:
        theta = np.zeros((x.shape[1], 1))

    a = np.dot(-np.array(y).T, np.log(hypothesis_function(x, theta)))
    b = np.dot((1 - np.array(y)).T, np.log(1 - hypothesis_function(x, theta)))
    c = (_lambda / 2) * np.sum(np.dot(theta[1:].T, theta[1:]))

    return (1.0 / y.size) * (np.sum(a - b) + c)


def optimize_theta(x, y, theta=None, _lambda=0):
    if theta is None:
        theta = np.zeros((x.shape[1], 1))

    return optimize.fmin(
        lambda _theta: compute_cost(x, y, _theta, _lambda),
        x0=theta,
        maxiter=400,
        full_output=True,
    )[0]


def predict(theta, score_1, score_2):
    return hypothesis_function(theta, np.array([1, score_1, score_2]))


def correct_predictions(theta, pos, neg):

    total = 0

    for score in pos:
        if predict(theta, score[1], score[2]) >= 0.5:
            total += 1

    for score in neg:
        if predict(theta, score[1], score[2]) < 0.5:
            total += 1

    return total / (len(pos) + len(neg))


def map_feature(x1_col, x2_col):
    degrees = 6
    result = np.ones((x1_col.shape[0], 1))

    for i in range(degrees):
        for j in range(i + 2):
            a = x1_col ** (i - j + 1)
            b = x2_col ** (j)
            result = np.hstack((result, (a * b).reshape(a.shape[0], 1)))
    return result


def optimize_regularized_theta(x, y, theta=None, _lambda=0):
    if theta is None:
        theta = np.zeros((x.shape[1], 1))

    result = optimize.minimize(
        lambda _theta: compute_cost(x, y, _theta, _lambda),
        theta,
        method="BFGS",
        options={"maxiter": 500, "disp": False},
    )
    return np.array([result.x])


def plot_data(data, labels, title="scatter_plot_of_training_data"):
    x, y = data
    label_pos, label_neg, label_x, label_y = labels
    plt.figure(figsize=(10, 6))
    plt.scatter(x[:, 1], x[:, 2], marker="+", c="k", s=150, label=label_pos)
    plt.scatter(y[:, 1], y[:, 2], marker="o", c="y", s=100, label=label_neg)
    plt.ylabel(label_y)
    plt.xlabel(label_x)
    plt.legend(loc="upper right")
    plt.savefig(title)


def plot_logistic_regression_fit(data):
    def fit(theta, x_range):
        return (-1.0 / theta[2]) * (theta[0] + theta[1] * x_range)

    theta, x, y, pos, neg = data
    x_range = np.array([np.min(x[:, 1]), np.max(x[:, 1])])

    plt.figure(figsize=(10, 6))
    plt.scatter(pos[:, 1], pos[:, 2], marker="+", c="k", s=150, label="Admitted")
    plt.scatter(neg[:, 1], neg[:, 2], marker="o", c="y", s=100, label="Not admitted")
    plt.plot(x_range, fit(theta, x_range), "b-")
    plt.ylabel("Exam 2 score")
    plt.xlabel("Exam 1 score")
    plt.legend(loc="upper right")
    plt.savefig("logistic_regression_fit")


def plot_boundary(data):

    theta, _lambda = data

    range_x = np.linspace(-1, 1.5, 50)
    range_y = np.linspace(-1, 1.5, 50)
    range_z = [
        [
            np.dot(theta, map_feature(np.array([y]), np.array([x])).T)[0][0]
            for x in range_x
        ]
        for y in range_y
    ]
    range_z = np.array(range_z).T

    contour = plt.contour(range_x, range_y, range_z, [0], cmap=plt.cm.coolwarm, extend="both")
    plt.clabel(contour, inline=1, fmt={0: f"Lambda = {_lambda}"})
    plt.title("Decision Boundary")


def part_1():
    x, y = read_data(DATA_PATH_1)
    pos = np.array([x[i] for i in range(len(x)) if y[i] == 1])
    neg = np.array([x[i] for i in range(len(x)) if y[i] == 0])
    plot_data((pos, neg), ("Admitted", "Not admitted", "Exam 1 score", "Exam 2 score"))
    # cost = compute_cost(x, y)

    theta = optimize_theta(x, y)
    plot_logistic_regression_fit((theta, x, y, pos, neg))

    with open("result.txt", "w") as output_file:
        output_file.write("Part 1:\n")
        output_file.write("Computed cost with the optimal parameters of theta:\n")
        output_file.write(f"{compute_cost(x, y, theta): .2f}\n")

        output_file.write(
            "\nFor a student with an Exam 1 score of 45 and an Exam 2 score of 85, computed admission probability is:\n"
        )
        output_file.write(f"{predict(theta, 45, 85): .2f}\n")

        output_file.write("\nCorrect samples in training data:\n")
        output_file.write(f"{correct_predictions(theta, pos, neg): .2f}\n")


def part_2():
    x, y = read_data(DATA_PATH_2)
    pos = np.array([x[i] for i in range(len(x)) if y[i] == 1])
    neg = np.array([x[i] for i in range(len(x)) if y[i] == 0])
    plot_data((pos, neg), ("y=1", "y=0", "Microchip Test 1", "Microchip Test 2"))
    x_mapped = map_feature(x[:, 1], x[:, 2])

    for _lambda in (1, 10, 100):
        theta = optimize_regularized_theta(x_mapped, y, _lambda=_lambda)
        plot_data(
            (pos, neg),
            ("y=1", "y=0", "Microchip Test 1", "Microchip Test 2"),
            f"decision_boundary_lambda_{_lambda}",
        )
        plot_boundary((theta, _lambda))

    with open("result.txt", "a") as output_file:
        output_file.write("\nPart 2:\n")
        output_file.write("Computed cost with mapped X:\n")
        output_file.write(f"{compute_cost(x_mapped, y): .2f}\n")


def main():
    plt.style.use("seaborn")
    part_1()
    part_2()
    plt.show()


if __name__ == "__main__":
    main()
