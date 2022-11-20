""""
The goal of this module is to implement all algorithms and numerical
methods needed to solve the Task 4 from the coding homeworks in the
Machine Learning course on coursera.com.
"""
from abc import ABC, abstractmethod

import numpy as np
from scipy import optimize
    

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


class NeuralNetwork(ModelBase):
    """
    Implementation of the neural network. The network includes input layer, single hidden layer
    and output.
    """

    def __init__(
        self,
        input_layer_size: int,
        hidden_layer_size: int,
        output_layer_size: int,
        _lambda: float = 0.0,
    ):
        """
        Initializes the neural network.

        :param input_layer_size: The size of the input layer.
        :param hidden_layer_size: The size of the hidden layer.
        :param output_layer_size: The size of the output layer.
        :param _lambda: The regularization parameter.
        """
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size
        self._lambda = _lambda

        self.theta_1 = np.random.rand(hidden_layer_size, input_layer_size + 1)
        self.theta_2 = np.random.rand(output_layer_size, hidden_layer_size + 1)

    def cost(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """
        Implementation of the cost function.

        :param x: The input data.
        :param y: The output data.
        :param regularization: The regularization parameter.
        :return: The cost function value.
        """
        m = x.shape[0]

        #     y_matrix = np.eye(10)[y.reshape(-1)]
        y_matrix = np.eye(10)[y]

        cost = 0

        for i in range(m):
            _, output_layer = self.propagate_forward(x[i])
            cost += np.sum(
                -y_matrix[i] * np.log(output_layer)
                - (1 - y_matrix[i]) * np.log(1 - output_layer)
            )

        cost /= m

        cost += (
            self._lambda
            / (2 * m)
            * (np.sum(self.theta_1[:, 1:] ** 2) + np.sum(self.theta_2[:, 1:] ** 2))
        )

        return cost

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

    def propagate_forward(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Propagates the input forward through the network.

        :param x: The input data.
        :return: The hidden layer and output layer.
        """
        hidden_layer = self.sigmoid(self.theta_1 @ x)
        hidden_layer = np.insert(hidden_layer, 0, 1)
        output_layer = self.sigmoid(self.theta_2 @ hidden_layer)
        return hidden_layer, output_layer

    def propagate_backward(
        self, x: np.ndarray, y: np.ndarray
    ) -> (np.ndarray, np.ndarray):
        """
        Propagates the input backward through the network.

        :param x: The input data.
        :param y: The output data.
        :return: The gradients for theta_1 and theta_2.
        """
        m = x.shape[0]

        y_matrix = np.eye(10)[y.reshape(-1)]

        delta_2_sum = np.zeros(self.theta_1.shape)
        delta_3_sum = np.zeros(self.theta_2.shape)

        for i in range(m):
            hidden_layer, output_layer = self.propagate_forward(x[i])
            delta_3 = output_layer - y_matrix[i]
            delta_2 = np.dot(delta_3, self.theta_2[:, 1:]) * self.sigmoid_gradient(
                hidden_layer[1:]
            )
            delta_2_sum += np.dot(delta_2.reshape(-1, 1), x[i].reshape(1, -1))
            delta_3_sum += np.dot(delta_3.reshape(-1, 1), hidden_layer.reshape(1, -1))

        delta_2_sum /= m
        delta_3_sum /= m

        delta_2_sum[:, 1:] += self._lambda / m * self.theta_1[:, 1:]
        delta_3_sum[:, 1:] += self._lambda / m * self.theta_2[:, 1:]

        return delta_2_sum, delta_3_sum

    def optimize_theta(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Optimizes the weights of the neural network.

        :param x: The input data.
        :param y: The output data.
        """
        theta = np.concatenate([self.theta_1.flatten(), self.theta_2.flatten()])

        def function_to_minimize(theta):
            self.theta_1 = theta[: self.theta_1.size].reshape(self.theta_1.shape)
            self.theta_2 = theta[self.theta_1.size :].reshape(self.theta_2.shape)
            return self.cost(x, y)

        def gradient_wrapper(theta):
            self.theta_1 = theta[: self.theta_1.size].reshape(self.theta_1.shape)
            self.theta_2 = theta[self.theta_1.size :].reshape(self.theta_2.shape)
            theta_1_grad, theta_2_grad = self.propagate_backward(x, y)
            return np.concatenate([theta_1_grad.flatten(), theta_2_grad.flatten()])

        result = optimize.minimize(
            fun=function_to_minimize,
            x0=theta,
            method="TNC",
            jac=gradient_wrapper,
            options={"maxiter": 1000},
        )

        self.theta_1 = result.x[
            : self.hidden_layer_size * (self.input_layer_size + 1)
        ].reshape(self.hidden_layer_size, self.input_layer_size + 1)
        self.theta_2 = result.x[
            self.hidden_layer_size * (self.input_layer_size + 1) :
        ].reshape(self.output_layer_size, self.hidden_layer_size + 1)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the model to the given data.

        :param x: The input data.
        :param y: The output data.
        """
        self.optimize_theta(x, y)

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
            return value if value != 9 else 0

        output = np.argmax(self.propagate_forward(row)[1])
        return map_to_correct_range(output)
