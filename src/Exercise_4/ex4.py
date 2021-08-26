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

    theta_1, theta_2 = data["Theta1"], data["Theta2"]
    theta_2 = np.roll(theta_2, 1, axis=0)

    return theta_1, theta_2


def clean_y(y):
    """
    Originally, 10 represented 0. Let's use 0 for 0.
    """

    y = y.ravel()

    for i, elem in enumerate(y):
        if elem == 10:
            y[i] = 0


def forward_propagation(x, theta):
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    theta_1, theta_2 = theta
    m = x.shape[0]

    # input layer a_1 is row from x matrix with added column of bias units (ones)
    a_1 = x

    # hidden layer computed with dot product of input layer and theta
    # plus column of bias units
    z_2 = np.dot(a_1, theta_1.T)
    a_2 = sigmoid(z_2)
    a_2 = np.concatenate([np.ones((m, 1)), a_2], axis=1)

    # output layer, computed like the hidden layer
    z_3 = np.dot(a_2, theta_2.T)
    a_3 = sigmoid(z_3)

    return ((a_1, a_2, a_3), (z_2, z_3))


def compute_cost(x, y, theta=None, _lambda=0):
    if theta is None:
        theta = np.zeros((x.shape[1], 1)).reshape(-1)

    theta_1, theta_2 = theta
    m = y.size

    y_matrix = np.eye(10)[y.reshape(-1)]

    j = -1 / m * (np.sum(y_matrix * np.log(x) + (1 - y_matrix) * np.log(1 - x)))
    reg = (
        _lambda
        / (2 * m)
        * (np.sum(np.square(theta_1[:, 1:])) + np.sum(np.square(theta_2[:, 1:])))
    )

    return j + reg


def backpropagation(a, z, y, theta, _lambda=0):
    def sigmoid_gradient(z):
        gradient = 1 / (1 + np.exp(-z))

        return gradient * (1 - gradient)

    a_1, a_2, a_3 = a
    z_2, z_3 = z
    theta_1, theta_2 = theta
    m = y.size

    y_matrix = np.eye(10)[y.reshape(-1)]

    d_3 = a_3 - y_matrix
    d_2 = np.dot(d_3, theta_2[:, 1:]) * sigmoid_gradient(z_2)

    theta_1_grad = 1 / m * np.dot(d_2.T, a_1)
    theta_2_grad = 1 / m * np.dot(d_3.T, a_2)

    theta_2_grad[:, 1:] += _lambda / m * theta_2[:, 1:]
    theta_1_grad[:, 1:] += _lambda / m * theta_1[:, 1:]

    return (theta_1_grad.ravel(), theta_2_grad.ravel())


def random_weights(num_input_connections, num_output_connections, epsilon=0.12):
    return (
        np.random.rand(num_output_connections, 1 + num_input_connections) * 2 * epsilon
        - epsilon
    )


def split_theta(theta, input_layer_size, hidden_layer_size, num_labels):
    theta_1 = np.reshape(
        theta[: hidden_layer_size * (input_layer_size + 1)],
        (hidden_layer_size, (input_layer_size + 1)),
    )

    theta_2 = np.reshape(
        theta[(hidden_layer_size * (input_layer_size + 1)) :],
        (num_labels, (hidden_layer_size + 1)),
    )

    return (theta_1, theta_2)


def find_optimized_theta(
    x,
    y,
    theta,
    input_layer_size,
    hidden_layer_size,
    num_labels,
    options={"maxiter": 100},
    _lambda=1,
):
    def f_wrapper(_theta):
        reshaped_theta = split_theta(
            _theta, input_layer_size, hidden_layer_size, num_labels
        )
        a, z = forward_propagation(x, reshaped_theta)
        cost = compute_cost(a[-1], y, reshaped_theta)
        gradient = backpropagation(a, z, y, reshaped_theta)
        return cost, np.concatenate([gradient[0].ravel(), gradient[1].ravel()])

    return optimize.minimize(
        f_wrapper, theta, jac=True, method="TNC", options=options
    ).x


def calculate_training_accuracy(x, y, theta):

    n = x.shape[0]
    correct = 0

    predictions = predict(x, theta)

    for prediction, expected in zip(predictions, y):
        if prediction == expected:
            correct += 1

    return correct / n


def predict(x, theta):
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    theta_1, theta_2 = theta
    m = x.shape[0]
    num_labels = theta_2.shape[0]

    h_1 = sigmoid(np.dot(np.concatenate([np.ones((m, 1)), x], axis=1), theta_1.T))
    h_2 = sigmoid(np.dot(np.concatenate([np.ones((m, 1)), h_1], axis=1), theta_2.T))

    return np.argmax(h_2, axis=1)


def display_random_grid(x, n=20, indices=None):

    if indices is None:
        indices = np.random.choice(x.shape[0], n)

    plt.figure(figsize=(6, 6))
    image = x[indices, 1:].reshape(-1, n).T
    plt.imshow(image)
    plt.axis("off")


def part_1():
    x, y = read_data(DATA_PATH_1)
    theta = read_weights(DATA_PATH_2)

    theta_1, theta_2 = theta
    theta = (theta_1, theta_2)
    display_random_grid(x)
    clean_y(y)
    a, z = forward_propagation(x, theta)

    with open("result.txt", "w") as output_file:
        output_file.write("Part 1:\n")
        output_file.write("Computed cost with theta loaded from weights.mat file:\n")
        output_file.write(f"{compute_cost(a[-1], y, theta): .2f}\n")

        output_file.write("Computed cost with regularization parameter (lambda=1):\n")
        output_file.write(f"{compute_cost(a[-1], y, theta, 1): .2f}\n")


def part_2():
    x, y = read_data(DATA_PATH_1)
    theta = read_weights(DATA_PATH_2)

    input_layer_size = 400
    hidden_layer_size = 25
    num_labels = 10

    theta_1 = random_weights(input_layer_size, hidden_layer_size)
    theta_2 = random_weights(hidden_layer_size, num_labels)

    theta = np.concatenate([theta_1.ravel(), theta_2.ravel()], axis=0)
    reshaped_theta = split_theta(theta, input_layer_size, hidden_layer_size, num_labels)

    clean_y(y)

    a, z = forward_propagation(x, reshaped_theta)
    gradient = backpropagation(a, z, y, reshaped_theta)

    theta = find_optimized_theta(
        x, y, theta, input_layer_size, hidden_layer_size, num_labels
    )
    reshaped_theta = split_theta(theta, input_layer_size, hidden_layer_size, num_labels)
    x = np.delete(x, 0, 1)
    accuracy = calculate_training_accuracy(x, y, reshaped_theta) * 100

    with open("result.txt", "a") as output_file:
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


def main():
    plt.style.use("seaborn")
    part_1()
    part_2()
    plt.show()


if __name__ == "__main__":
    main()
