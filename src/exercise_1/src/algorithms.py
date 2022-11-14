""""
The goal of this module is to implement all algorithms and numerical
methods needed to solve the Task 1 from the coding homeworks in the
Machine Learning course on coursera.com.
"""
from abc import ABC, abstractmethod

import numpy as np


class LinearRegressionBase(ABC):
    """
    Base class for linear regression models.
    """

    def __init__(self):
        """
        Initializes the model.
        """
        self.theta = None

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the model to the data.

        :param x: The input data.
        :param y: The output data.
        """
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts the output for the given input.

        :param x: The input data.
        :return: The predicted output.
        """
        pass

    def __str__(self) -> str:
        """ "
        Returns a string representation of the model
        in the form of a linear equation.

        :return: A string representation of the model.
        """
        return f"y = {self.theta[0]:.2f} + {self.theta[1]:.2f}x"


class LinearRegressionGD(LinearRegressionBase):
    """ "
    The goal of this class is to implement a linear regression model.
    The gradient descent algorithm is used to learn the parameters of the model.
    """

    def __init__(self):
        super().__init__()
        self.theta_history = []
        self.cost_history = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """ "
        Fits the model to the data using gradient descent algorithm.

        :param x: Input dataset. A matrix with m rows and n columns.
        :param y: The values corresponding to the input dataset. An m-element vector.
        """
        m, n = x.shape
        self.theta = np.ones(n)
        alpha = 0.01
        num_iters = 1500

        for _ in range(num_iters):
            hypothesis = x @ self.theta
            loss = hypothesis - y
            gradient = x.T @ loss / m
            self.theta -= alpha * gradient
            self.theta_history.append(self.theta)
            self.cost_history.append(self.cost(x, y))

    def cost(self, x: np.ndarray, y: np.ndarray) -> float:
        """ "
        Computes the cost function for the given data and the current theta values.

        :param x: Input dataset. A matrix with m rows and n columns.
        :param y: The values corresponding to the input dataset. An m-element vector.
        :return: The value of the cost function.
        """
        m = x.shape[0]
        hypothesis = x @ self.theta
        square_loss = (hypothesis - y) ** 2

        return 1 / (2 * m) * np.sum(square_loss)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """ "
        Predicts the output for the given input.

        :param x: Input dataset. A matrix with m rows and n columns.
        :return: Predicted values for the given input.
        """
        return x @ self.theta

    def copy(self) -> "LinearRegressionGD":
        """ "
        Creates a copy of the model.

        :return: A copy of the model.
        """
        model = LinearRegressionGD()
        model.theta = self.theta.copy()
        model.theta_history = self.theta_history.copy()
        model.cost_history = self.cost_history.copy()

        return model


class LinearRegressionNE(LinearRegressionBase):
    """ "
    The goal of this class is to implement a linear regression model.
    The normal equation algorithm is used to learn the parameters of the model.
    """

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """ "
        Fits the model to the data using normal equation algorithm.

        :param x: Input dataset. A matrix with m rows and n columns.
        :param y: The values corresponding to the input dataset. An m-element vector.
        """
        self.theta = np.linalg.inv(x.T @ x) @ x.T @ y

    def predict(self, x: np.ndarray) -> np.ndarray:
        """ "
        Predicts the output for the given input.

        :param x: Input dataset. A matrix with m rows and n columns.
        :return: Predicted values for the given input.
        """
        return x @ self.theta

    def copy(self) -> "LinearRegressionNE":
        """ "
        Creates a copy of the model.

        :return: A copy of the model.
        """
        model = LinearRegressionNE()
        model.theta = self.theta.copy()

        return model
