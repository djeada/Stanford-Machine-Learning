import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import scipy.io

DATA_PATH_1 = "data/data.mat"
DATA_PATH_2 = "data/weights.mat"


def read_data(path):
    """
    x is a matrix with m rows and n columns
    y is a matrix with m rows and 1 column
    """

    data = scipy.io.loadmat(path)
    x, y = data["X"], data["y"]
    ones = np.ones(len(x))
    # prepend column of ones
    x = np.insert(x, 0, ones, axis=1)

    return x, y


def read_weights(path):
    data = scipy.io.loadmat(path)

    return data["Theta1"], data["Theta2"]


def hypothesis_function(x, theta):
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    return sigmoid(np.dot(x, theta))


def compute_cost(x, y, theta=None, _lambda=0):
    if theta is None:
        theta = np.zeros((x.shape[1], 1)).reshape(-1)

    m = y.size

    a = np.log(hypothesis_function(x, theta)).dot(-y.T)
    b = np.log(1 - hypothesis_function(x, theta)).dot(1 - y.T)
    return 1 / (2 * m) * ((a - b) * 2 + theta.T.dot(theta) * _lambda)


def compute_gradient(x, y, theta=None, _lambda=0):
    if theta is None:
        theta = np.zeros((x.shape[1], 1)).reshape(-1)

    m = y.size
    grad = 1 / m * np.dot(x.T, hypothesis_function(x, theta) - y.T)
    grad[1:] += theta[1:] * (_lambda / m)

    return grad


def optimize_theta(x, y, theta=None, _lambda=0):
    if theta is None:
        theta = np.zeros((x.shape[1], 1)).reshape(-1)

    return optimize.fmin_cg(
        lambda _theta: compute_cost(x, y, _theta, _lambda),
        x0=theta,
        fprime=lambda _theta: compute_gradient(x, y, _theta, _lambda),
        maxiter=50,
        disp=False,
        full_output=True,
    )[0]


def train_model(x, y):

    theta = list()

    for digit in range(1, 11):
        print(f"Optimizing digit {digit % 10}")
        theta.append(
            optimize_theta(x, np.array([1 if elem[0] == digit else 0 for elem in y]))
        )

    return np.array(theta)


def predict_one_vs_all(row, theta):
    hypothesis = np.array([hypothesis_function(row, theta[i]) for i in range(10)])
    return np.argmax(hypothesis)


def calculate_training_accuracy(x, y, theta, prediction_function):

    n = x.shape[0]
    correct = 0

    for i in range(n):
        prediction = prediction_function(x[i], theta) + 1
        if prediction == y[i]:
            correct += 1

    return correct / n


def propagate_forward(row, theta):
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    theta_1, theta_2 = theta

    hidden_layer = sigmoid(np.dot(row, theta_1.T))
    hidden_layer = np.concatenate([np.ones([1]), hidden_layer])

    return sigmoid(np.dot(hidden_layer, theta_2.T))


def predict_nn(row, theta):
    return np.argmax(propagate_forward(row, theta))


def display_random_grid(x, n=20, indices=None):

    if indices is None:
        indices = np.random.choice(x.shape[0], n)

    plt.figure(figsize=(6, 6))
    image = x[indices, 1:].reshape(-1, n).T
    plt.imshow(image)
    plt.axis("off")


def part_1():
    x, y = read_data(DATA_PATH_1)
    display_random_grid(x)
    theta = train_model(x, y)
    accuracy = calculate_training_accuracy(x, y, theta, predict_one_vs_all) * 100

    with open("result.txt", "w") as output_file:
        output_file.write("Part 1:\n")
        output_file.write("Training set accuracy:\n")
        output_file.write(f"{accuracy: .2f}\n")


def part_2():
    x, y = read_data(DATA_PATH_1)
    theta = read_weights(DATA_PATH_2)

    accuracy = calculate_training_accuracy(x, y, theta, predict_nn) * 100

    with open("result.txt", "a") as output_file:
        output_file.write("Part 2:\n")
        output_file.write("Training set accuracy:\n")
        output_file.write(f"{accuracy: .2f}\n")


def main():
    plt.style.use("seaborn")
    part_1()
    part_2()
    plt.show()


if __name__ == "__main__":
    main()
