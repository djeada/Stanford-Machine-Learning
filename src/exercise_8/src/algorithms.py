""""
The goal of this module is to implement all algorithms and numerical
methods needed to solve the Task 8 from the coding homeworks in the
Machine Learning course on coursera.com.
"""

import numpy as np
import scipy.stats as stats
from scipy import optimize
from typing import Tuple, List


def compute_f1(predictions: np.ndarray, real_values: np.ndarray) -> float:
    """
    Calculates the F1 score, using the formula:
    F1 = 2 * (precision * recall) / (precision + recall)

    :param predictions: The predictions.
    :param real_values: The real values.
    :return: The F1 score.
    """

    tp = np.sum((predictions == 1) & (real_values == 1))
    fp = np.sum((predictions == 1) & (real_values == 0))
    fn = np.sum((predictions == 0) & (real_values == 1))

    p = tp / (tp + fp) if tp + fp > 0 else 0
    r = tp / (tp + fn) if tp + fn > 0 else 0

    return 2 * p * r / (p + r) if p + r > 0 else 0


class GaussianRegression:
    """
    Gaussian regression model.

    :param mu: The mean of the Gaussian distribution.
    :param sigma_2: The variance of the Gaussian distribution.
    """

    def __init__(self, mu: np.ndarray, sigma_2: np.ndarray):
        self.mu = mu
        self.sigma_2 = sigma_2
        self.epsilon = None

    @classmethod
    def from_array(cls, x: np.ndarray) -> "GaussianRegression":
        """
        Creates a GaussianRegression object from an array of data.

        :param x: The data array.
        :return: The GaussianRegression object.
        """
        mu, sigma_2 = np.mean(x, axis=0), np.var(x, axis=0)
        return cls(mu, sigma_2)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts the probability of a given data point.

        :param x: The data point.
        :return: The probability.
        """
        return stats.multivariate_normal.pdf(x=x, mean=self.mu, cov=self.sigma_2)

    def select_threshold(self, x: np.ndarray, y: np.ndarray) -> tuple:
        """
        Computes the best threshold (epsilon) to use for selecting outliers.

        :param x: The validation set.
        :param y: The labels of the validation set.
        :return: The best threshold and the corresponding F1 score.
        """
        x_cv = self.predict(x)
        epsilons = np.linspace(np.min(x_cv), np.max(x_cv), 1000)

        best_f1, best_eps = 0, 0
        real_values = (y == 1).flatten()
        for epsilon in epsilons:
            predictions = x_cv < epsilon
            f1 = compute_f1(predictions, real_values)
            if f1 > best_f1:
                best_f1, best_eps = f1, epsilon

        self.epsilon = best_eps
        return best_eps, best_f1

    def is_anomaly(self, x: np.ndarray) -> bool:
        """
        Checks if a given data point is an anomaly.

        :param x: The data point.
        :return: True if the data point is an anomaly, False otherwise.
        """
        if self.epsilon is None:
            raise ValueError("Please select a threshold first.")
        return self.predict(x) < self.epsilon


class CollaborativeFilteringRegression:
    """
    Collaborative filtering regression model.

    :param _lambda: The regularization parameter.
    """

    def __init__(
        self,
        _lambda: float = 1.0,
    ):
        self.theta = None
        self._lambda = _lambda

    def cost(self, x: np.ndarray, y: np.ndarray, r: np.ndarray) -> float:
        """
        Computes the cost of the model.

        :param x: The feature matrix.
        :param y: The rating matrix.
        :param r: The rating matrix.
        :return: The cost.
        """

        cost = 0.5 * np.sum(((x @ self.theta.T - y) * r) ** 2)
        cost += 0.5 * self._lambda * np.sum(self.theta ** 2)
        cost += 0.5 * self._lambda * np.sum(x ** 2)

        return cost

    def gradient(self, x: np.ndarray, y: np.ndarray, r: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the model.

        :param x: The feature matrix.
        :param y: The rating matrix.
        :param r: The rating matrix.
        :return: The gradient.
        """
        x_grad = ((x @ self.theta.T - y) * r) @ self.theta
        x_grad += self._lambda * x

        theta_grad = ((x @ self.theta.T - y) * r).T @ x
        theta_grad += self._lambda * self.theta

        return np.concatenate((x_grad.flatten(), theta_grad.flatten()))

    def fit(self, x: np.ndarray, y: np.ndarray, r: np.ndarray) -> None:
        """
        Fits the model.

        :param x: The feature matrix.
        :param y: The rating matrix.
        :param r: The rating matrix.
        """

        x = np.random.rand(5, 3)
        theta = np.random.rand(4, 3)
        params = np.concatenate((x.flatten(), theta.flatten()))

        def cost_wrapper(params):
            x = params[:15].reshape(5, 3)
            self.theta = params[15:].reshape(4, 3)
            return self.cost(x, y, r)

        def gradient_wrapper(params):
            x = params[:15].reshape(5, 3)
            self.theta = params[15:].reshape(4, 3)
            return self.gradient(x, y, r)

        res = optimize.minimize(
            cost_wrapper,
            params,
            method="CG",
            jac=gradient_wrapper,
            options={"maxiter": 100},
        )

        self.theta = res.x[15:].reshape(4, 3)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts the ratings of a given user.

        :param x: The feature matrix.
        :return: The predicted ratings.
        """
        return x @ self.theta.T

    def top_predictions(
        self, x: np.ndarray, n: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts the top n recommendations for a given user.

        :param x: The feature matrix.
        :param n: The number of recommendations.
        :return: The top n recommendations and their indices.
        """
        predictions = self.predict(x)
        indices = np.argsort(predictions, axis=1)[:, -n:]
        return predictions[0, indices[0]], indices[0]
