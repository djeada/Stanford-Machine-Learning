""""
The goal of this module is to implement all algorithms and numerical
methods needed to solve the Task 5 from the coding homeworks in the
Machine Learning course on coursera.com.
"""

from typing import Tuple
import numpy as np
import scipy.optimize


class LinearRegression:
    """ "
    The goal of this class is to implement a linear regression model.
    The gradient descent algorithm is used to learn the parameters of the model.
    """

    def __init__(self, _lambda: float = 0.0):
        self._lambda = _lambda
        self.theta = []
        self.theta_history = []
        self.cost_history = []

    def hypothesis_function(self, x: np.ndarray) -> np.ndarray:
        """
        Takes an x matrix and returns the hypothesis function for the current theta.
        """
        return x @ self.theta

    def compute_cost(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Takes an x matrix and a y matrix and returns the cost function for the current theta.
        """
        m = x.shape[0]

        j = (
            1
            / (2 * m)
            * np.dot(
                (self.hypothesis_function(x).reshape((m, 1)) - y).T,
                (self.hypothesis_function(x).reshape((m, 1)) - y),
            )
        )
        reg = self._lambda / (2 * m) * np.dot(self.theta[1:].T, self.theta[1:])

        return (j + reg)[0][0]

    def compute_cost_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Takes an x matrix and a y matrix and returns the gradient of the cost function for the current theta.
        """
        m = x.shape[0]

        grad = (
            1
            / m
            * np.dot(
                (self.hypothesis_function(x).reshape((m, 1)) - y).T,
                x,
            ).T
        )

        reg = self._lambda / m * self.theta

        reg[0] = 0

        return grad + reg

    def optimize_theta(self, x: np.ndarray, y: np.ndarray) -> None:

        """
        Takes an x matrix and a y matrix and optimizes the theta using scipy.optimize.minimize.
        """

        def function_to_minimize(theta):
            self.theta = theta
            return self.compute_cost(x, y)

        result = scipy.optimize.minimize(
            function_to_minimize,
            x0=self.theta,
            method="BFGS",
            options={
                "disp": False,
                "gtol": 1e-05,
                "eps": 1.4901161193847656e-08,
                "return_all": False,
                "maxiter": None,
            },
        ).x
        return result

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Takes an x matrix and a y matrix and fits the model to the data.
        """
        self.theta = np.zeros((x.shape[1], 1))
        result = self.optimize_theta(x, y)
        self.theta = result

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Takes an x matrix and returns the predicted y values.
        """
        return self.hypothesis_function(x)

    def __str__(self) -> str:
        """ "
        Returns a string representation of the model
        in the form of a linear equation.

        :return: A string representation of the model.
        """
        return f"y = {self.theta[0]:.2f} + {self.theta[1]:.2f}x"


class FeatureNormalizer:
    """
    A class that normalizes features by subtracting the mean and dividing by the standard deviation.
    """

    def __init__(self, x: np.ndarray, excluded_columns: Tuple[int, ...] = ()):
        """
        Initializes a new instance of the FeatureNormalizer class.
        :param x: The input array.
        :param excluded_columns: The columns to exclude from normalization.
        """
        self.means = np.mean(x, axis=0)
        self.stds = np.std(x, axis=0)

        for i in excluded_columns:
            self.means[i] = 0
            self.stds[i] = 1

    def normalize(self, x: np.ndarray, epsilon=1e-100) -> np.ndarray:
        """
        Normalizes the input array.
        :param x: The input array.
        :param epsilon: A small value to avoid division by zero.
        :return: The normalized input array.
        """
        return (x - self.means) / (self.stds + epsilon)

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """
        Denormalizes the input array.
        :param x: The input array.
        :return: The denormalized input array.
        """
        return x * self.stds + self.means


class PolynomialRegression(LinearRegression):
    """
    The goal of this class is to implement a polynomial regression model.
    Polynomial regression is similar to linear regression, but the features
    are the powers of the original features up to the p-th power. This fact
    has to be taken into account when implementing the methods.
    For the supplied data x, we have to disregard first column of ones.
    Then construct polynomial features from it and then normalize it.
    """

    def __init__(self, p: int, _lambda: float = 0):
        """
        Initializes a new instance of the PolynomialRegression class.
        :param p: The degree of the polynomial.
        :param _lambda: The regularization parameter.
        """
        super().__init__(_lambda)
        self.p = p
        self.feature_normalizer = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Takes an x matrix and a y matrix and fits the model to the data.
        """
        x = x[:, 1:]
        x = self.construct_polynomial_features(x)
        self.feature_normalizer = FeatureNormalizer(x)
        x = self.feature_normalizer.normalize(x)
        x = np.hstack((np.ones((x.shape[0], 1)), x))
        super().fit(x, y)

    def process_input(self, x: np.ndarray) -> np.ndarray:
        """
        Takes an x matrix and returns the processed x matrix.
        """
        x = np.atleast_2d(x)
        x = x[:, 1:]
        x = self.construct_polynomial_features(x)
        x = self.feature_normalizer.normalize(x)
        x = np.hstack((np.ones((x.shape[0], 1)), x))
        return x

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Takes an x matrix and returns the predicted y values.
        """
        x = self.process_input(x)
        return super().predict(x)

    def construct_polynomial_features(self, x: np.ndarray) -> np.ndarray:
        """
        Takes an x matrix and returns the polynomial features.
        """
        x = np.hstack([x ** i for i in range(1, self.p + 1)])
        return x

    def __str__(self) -> str:
        """ "
        Returns a string representation of the model
        in the form of a polynomial equation.

        :return: A string representation of the model.
        """
        s = "y = "
        for i in range(self.p + 1):
            if i == 0:
                s += f"{self.theta[i]:.2f}"
            elif i == 1:
                s += f" + {self.theta[i]:.2f}x"
            else:
                s += f" + {self.theta[i]:.2f}x^{i}"
        return s
