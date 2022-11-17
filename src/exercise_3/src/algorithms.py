""""
The goal of this module is to implement all algorithms and numerical
methods needed to solve the Task 3 from the coding homeworks in the
Machine Learning course on coursera.com.
"""
from abc import ABC, abstractmethod

import numpy as np
from scipy import optimize
from typing import Callable


class ModelBase(ABC):
    """
    Base class for all models.
    """

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the model to the given data.

        :param x: The input data.
        :param y: The output data.
        """

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts the output data for the given input data.

        :param x: The input data.
        :return: The predicted output data.
        """


class MulticlassLogisticRegression(ModelBase):
    """
    Logistic regression with support for multiple classes.
    """

    def __init__(self, _lambda: int = 0):
        """
        Initializes the class.

        :param _lambda: The regularization parameter.
        """
        self.thetas = []  # thetas for each class
        self.cost_history = []  # cost history for each class
        self._lambda = _lambda

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the model to the data.

        :param x: The input data.
        :param y: The output data.
        """
        unique_labels = np.unique(y)
        n = len(unique_labels)
        self.thetas = np.zeros((n, x.shape[1]))  # reset thetas
        self.cost_history = [[] for _ in range(n)]  # reset cost history

        for i, label in enumerate(unique_labels):
            print(f"Optimizing digit {label}")
            current_class = np.array([1 if elem[0] == label else 0 for elem in y])
            self.optimize_theta(x, current_class, i)

    def hypothesis_function(self, x: np.ndarray, theta_index: int = -1) -> np.ndarray:
        """
        Returns the output of applying the hypothesis function to the input for
        each theta. If theta_index is specified, returns the output of applying
        the hypothesis function to the input for the specified theta.

        :param x: The input data.
        :param theta_index: The index of the theta.
        :return: The hypothesis function output.
        """

        def sigmoid(x: np.ndarray) -> np.ndarray:
            """
            Implements the sigmoid function.

            :param x: The input data.
            :return: The output of the sigmoid function for each element in x.
            """
            return 1 / (1 + np.exp(-x))

        if theta_index == -1:
            return np.array([sigmoid(x @ theta) for theta in self.thetas])

        return sigmoid(np.dot(x, self.thetas[theta_index]))

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Returns the prediction.

        :param x: The input data.
        :return: The prediction.
        """
        return np.argmax(self.hypothesis_function(x), axis=0)

    def compute_cost(self, x: np.ndarray, y: np.ndarray, theta_index: int) -> float:
        """
        Returns the cost function.

        :param x: The input data.
        :param y: The output data.
        :param theta_index: The index of the theta.
        :return: The cost function.
        """
        theta = self.thetas[theta_index]
        m = y.size

        hypothesis_output = self.hypothesis_function(x, theta_index)
        cost = (
            1
            / m
            * np.sum(
                -y.T @ np.log(hypothesis_output)
                - (1 - y.T) @ np.log(1 - hypothesis_output)
            )
        )
        cost += self._lambda / (2 * m) * np.sum(theta[1:] ** 2)  # regularization

        return cost

    def compute_gradient(
        self, x: np.ndarray, y: np.ndarray, theta_index: int
    ) -> np.ndarray:
        """
        Returns the gradient function.

        :param x: The input data.
        :param y: The output data.
        :param theta_index: The index of the theta.
        :return: The gradient function.
        """
        theta = self.thetas[theta_index]

        m = y.size
        hypothesis_output = self.hypothesis_function(x, theta_index)
        gradient = 1 / m * (hypothesis_output - y) @ x

        # regularization
        gradient[1:] += self._lambda / m * theta[1:]

        return gradient

    def optimize_theta(self, x: np.ndarray, y: np.ndarray, theta_index: int) -> None:
        """
        Returns the optimized value of theta.

        :param x: The input data.
        :param y: The output data.
        :param theta_index: The index of the theta to optimize.
        """
        theta = self.thetas[theta_index]

        def function_to_minimize(theta: np.ndarray) -> float:
            """
            Returns the value of the cost function for the given theta.

            :param theta: The theta.
            :return: The value of the cost function.
            """
            self.thetas[theta_index] = theta
            cost = self.compute_cost(x, y, theta_index)
            self.cost_history[theta_index].append(cost)
            return cost

        result = optimize.fmin_cg(
            function_to_minimize,
            x0=theta,
            fprime=lambda _: self.compute_gradient(x, y, theta_index),
            maxiter=50,
            disp=False,
            full_output=True,
        )[0]

        self.thetas[theta_index] = result


class NeuralNetwork(ModelBase):
    """
    Implementation of the neural network. The network includes input layer, single hidden layer
    and output.
    """

    def __init__(
        self, input_layer_size: int, hidden_layer_size: int, output_layer_size: int
    ):
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size

        self.theta_1 = np.random.rand(hidden_layer_size, input_layer_size + 1)
        self.theta_2 = np.random.rand(output_layer_size, hidden_layer_size + 1)

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Sigmoid function implementation. It will apply the sigmoid function
        to each member of a given vector z.

        :param z: The input vector.
        :return: The output vector.
        """
        return 1 / (1 + np.exp(-z))

    def sigmoid_gradient(self, z: np.ndarray) -> np.ndarray:
        """
        Sigmoid gradient function implementation. It will apply the sigmoid gradient function
        to each member of a given vector z.

        :param z: An n+1-element vector.
        :return: A vector of sigmoid gradient function values for every element of z.
        """
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def propagate_forward(self, row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Implementation of the forward propagation algorithm.

        :param row: The input data.
        :return: The output of the forward propagation algorithm.
        """
        hidden_layer = self.sigmoid(np.dot(row, self.theta_1.T))
        hidden_layer = np.concatenate([np.ones([1]), hidden_layer])

        output_layer = self.sigmoid(np.dot(hidden_layer, self.theta_2.T))

        return hidden_layer, output_layer

    def propagate_backward(
        self,
        row: np.ndarray,
        y: np.ndarray,
        hidden_layer: np.ndarray,
        output_layer: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Implementation of the backward propagation algorithm.

        :param row: A single row form the matrix X.
        :param y: A single row form the matrix Y.
        :param hidden_layer: A vector representing the hidden layer.
        :param output_layer: A vector representing the output layer.
        :return: Two vectors representing the gradient of theta_1 and theta_2.
        """

        def add_bias_unit(row: np.ndarray) -> np.ndarray:
            """
            Adds a bias unit to a given row.

            :param row: The row to add bias unit.
            :return: The row with bias unit.
            """
            return np.concatenate([np.ones([1]), row])

        y = add_bias_unit(y)

        delta_3 = output_layer - y
        delta_2 = np.dot(delta_3, self.theta_2) * self.sigmoid_gradient(hidden_layer)

        delta_2 = delta_2[1:]

        theta_1_grad = np.dot(delta_2.T, add_bias_unit(row))
        theta_2_grad = np.dot(delta_3.T, hidden_layer)

        return theta_1_grad, theta_2_grad

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 0.1,
        iterations: int = 1000,
        regularization: float = 0.0,
    ) -> None:
        """
        Trains the neural network.

        :param x: The input data.
        :param y: The output data.
        :param learning_rate: The learning rate.
        :param iterations: The number of iterations.
        :param regularization: The regularization parameter.
        """

        m = x.shape[0]

        for _ in range(iterations):
            for i in range(m):
                hidden_layer, output_layer = self.propagate_forward(x[i])
                theta_1_grad, theta_2_grad = self.propagate_backward(
                    x[i], y[i], hidden_layer, output_layer
                )

                self.theta_1 -= learning_rate * (
                    theta_1_grad + regularization * self.theta_1
                )
                self.theta_2 -= learning_rate * (
                    theta_2_grad + regularization * self.theta_2
                )

    def predict(self, row: np.ndarray) -> float:
        """
        Using a trained neural network, makes a prediction for a given input.

        :param row: A single row form the matrix X.
        :return: A prediction for the given input.
        """

        def map_to_correct_range(value: float) -> float:
            """
            Maps the value to 0-9 range.

            :param value: The value to be mapped.
            :return: The mapped value.
            """
            return value + 1 if value != 9 else 0

        output = np.argmax(self.propagate_forward(row)[1])
        return map_to_correct_range(output)
