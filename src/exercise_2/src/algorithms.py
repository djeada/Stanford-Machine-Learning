""""
The goal of this module is to implement all algorithms and numerical
methods needed to solve the Task 2 from the coding homeworks in the
Machine Learning course on coursera.com.
"""

import numpy as np
from scipy import optimize


class LogisticRegression:
    """
    Logistic Regression class.
    """

    def __init__(self, _lambda: int = 0):
        self._lambda = _lambda
        self.theta = None
        self.cost_history = []

    def hypothesis_function(self, x: np.ndarray) -> np.ndarray:
        """
        Logistic hypothesis function.

        Args:
          x:
            Input dataset. A matrix with m rows and n columns.
          theta:
            The parameters for the regression function. An n+1-element vector.

        Returns:
          A vector of values calculated using the logistic hypothesis function.
        """

        def sigmoid(z: np.ndarray) -> np.ndarray:
            """
            Sigmoid function implementation. It will apply the sigmoid function
            to each member of a given vector z.

            Args:
              z:
                An n+1-element vector.

            Returns:
                A vector of sigmoid function values for every element of z.
            """
            return 1 / (1 + np.exp(-z))

        return sigmoid(x @ self.theta)

    def cost(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the cost of using theta as the parameter for logistic regression.
        Uses the current value of theta and the data in x and y.

        :param x: Input dataset. A matrix with m rows and n columns.
        :param y: The values corresponding to the input dataset. An m-element vector.
        :return: The cost of using theta as the parameter for logistic regression.
        """
        m = x.shape[0]
        hypothesis_output = self.hypothesis_function(x)
        cost = (
            -1
            / m
            * (
                y.T @ np.log(hypothesis_output)
                + (1 - y).T @ np.log(1 - hypothesis_output)
            )
        )
        return cost

    def fit(self, x: np.ndarray, y: np.ndarray, _lambda: int = 0) -> None:
        """
        Fits the model to the data.

        :param x: Input dataset. A matrix with m rows and n columns.
        :param y: The values corresponding to the input dataset. An m-element
        vector.
        :param _lambda: For regularization, non-zero lambda values are utilized.
        There is no regularization when lambda is set to 0.
        """
        self.theta = np.zeros((x.shape[1], 1))
        self.cost_history = []

        def function_to_minimize(theta):
            self.theta = theta.reshape(-1, 1)
            cost = self.cost(x, y)
            self.cost_history.append(cost)
            return cost

        self.theta = optimize.fmin(
            function_to_minimize,
            np.zeros(x.shape[1]),
            maxiter=400,
            full_output=True,
        )[0]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Using the hypothesis function, make a prediction. The logistic function
        requires an n+1-element vector and three values to be used as parameters.
        The three parameters are: 1, score 1, and score 2.

        :param x: Input dataset. A matrix with m rows and n columns.
        :return: The value returned by the hypothesis function.
        """
        return self.hypothesis_function(x) >= 0.5

    def boundary(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the decision boundary.

        :param x: Input dataset. A matrix with m rows and n columns.
        :return: The decision boundary.
        """
        return (-1 / self.theta[2]) * (self.theta[1] * x + self.theta[0])

    def __str__(self):
        """
        String representation of the class.

        :return: String representation of the class.
        """
        return (
            f"{self.theta[0]:.2f} + {self.theta[1]:.2f}x1 + {self.theta[2]:.2f}x2 = 0"
        )


class RegularizedLogisticRegression(LogisticRegression):
    """
    Regularized Logistic Regression class.
    """

    def __init__(self, _lambda: int = 0):
        super().__init__(_lambda)

    def cost(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the cost of using theta as the parameter for logistic regression.
        Uses the current value of theta and the data in x and y.

        :param x: Input dataset. A matrix with m rows and n columns.
        :param y: The values corresponding to the input dataset. An m-element vector.
        :return: The cost of using theta as the parameter for logistic regression.
        """
        m = x.shape[0]
        hypothesis_output = self.hypothesis_function(x)
        cost = (
            -1
            / m
            * (
                y.T @ np.log(hypothesis_output)
                + (1 - y).T @ np.log(1 - hypothesis_output)
            )
        )
        cost += self._lambda / (2 * m) * np.sum(self.theta[1:] ** 2)
        return cost

    def fit(self, x: np.ndarray, y: np.ndarray, _lambda: int = 0) -> None:
        """
        Fits the model to the data.

        :param x: Input dataset. A matrix with m rows and n columns.
        :param y: The values corresponding to the input dataset. An m-element
        vector.
        :param _lambda: For regularization, non-zero lambda values are utilized.
        There is no regularization when lambda is set to 0.
        """
        self.theta = np.zeros((x.shape[1], 1))
        self.cost_history = []

        def function_to_minimize(theta):
            self.theta = theta.reshape(-1, 1)
            cost = self.cost(x, y)
            self.cost_history.append(cost)
            return cost

        self.theta = optimize.fmin(
            function_to_minimize,
            np.zeros(x.shape[1]),
            maxiter=400,
            full_output=True,
        )[0]
