""""
The goal of this module is to implement all algorithms and numerical
methods needed to solve the Task 6 from the coding homeworks in the
Machine Learning course on coursera.com.
"""
from typing import List

import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod


class SvmRegressionBase(ABC):
    """ "
    The interface for the SVM regression models.
    """

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Train the SVM regression model.

        :param x: The input data.
        :param y: The output data.
        """
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the output of the SVM regression model.

        :param x: The data to predict.
        """
        pass

    @abstractmethod
    def boundary(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the boundary of the SVM regression model.

        :param x: The input data.
        """
        pass

    @abstractmethod
    def __str__(self):
        """
        String representation of the class.
        :return: String representation of the class.
        """
        pass


class LinearSvmRegression(SvmRegressionBase):
    """
    Support Vector Machine Regression using linear kernel.
    """

    def __init__(self, c: float = 1.0) -> None:
        """
        Initialize the SVM regression model.

        :param c: The regularization parameter.
        """
        self.c = c
        self.model = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Train the SVM regression model.

        :param x: The input data.
        :param y: The output data.
        """
        self.model = svm.SVC(C=self.c, kernel="linear")
        self.model.fit(x, y.flatten())

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the output of the SVM regression model.

        :param x: The data to predict.
        """
        x = np.atleast_2d(x)
        return self.model.predict(x)

    def boundary(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the boundary of the SVM regression model.

        :param x: The input data.
        """
        w = self.model.coef_[0]
        a = -w[0] / w[1]
        b = -self.model.intercept_[0] / w[1]
        return a * x + b

    def __str__(self):
        """
        String representation of the class.
        :return: String representation of the class.
        """
        w = self.model.coef_[0]
        a = -w[0] / w[1]
        b = -self.model.intercept_[0] / w[1]
        return f"y = {a:.2f} * x + {b:.2f}"


class GaussianSvmRegression(SvmRegressionBase):
    """
    Support Vector Machine Regression using Gaussian kernel.
    """

    def __init__(self, c: float = 1.0, sigma: float = 0.1) -> None:
        """
        Initialize the SVM regression model.

        :param c: The regularization parameter.
        :param sigma: The sigma of the Gaussian kernel.
        """
        self.c = c
        self.sigma = sigma
        self.model = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Train the SVM regression model.

        :param x: The input data.
        :param y: The output data.
        """
        self.model = svm.SVC(C=self.c, kernel="rbf", gamma=1 / (2 * self.sigma ** 2))
        self.model.fit(x, y.flatten())

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the output of the SVM regression model.

        :param x: The data to predict.
        """
        x = np.atleast_2d(x)
        return self.model.predict(x)

    def boundary(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the boundary of the SVM regression model.

        :param x: The input data.
        """
        x = np.atleast_2d(x)
        return self.model.decision_function(x)

    def __str__(self):
        """
        String representation of the class.
        :return: String representation of the class.
        """
        return f"Gaussian SVM with sigma = {self.sigma}"


class GaussianSvmRegressionCrossValidation:
    """
    Cross validation for the Gaussian SVM regression model.
    """

    def __init__(self, possible_c: List[float], possible_sigma: List[float]) -> None:
        """
        Initialize the cross validation.

        :param possible_c: The possible values of the regularization parameter.
        :param possible_sigma: The possible values of the sigma of the Gaussian kernel.
        """
        self.possible_c = possible_c
        self.possible_sigma = possible_sigma

    def fit(self, x: np.ndarray, y: np.ndarray) -> GaussianSvmRegression:
        """
        Train the SVM regression model.

        :param x: The input data.
        :param y: The output data.
        :return: The best SVM regression model.
        """
        # randomly pick 0.2 of the data for validation
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
        best_score = 0
        best_model = None
        for c in self.possible_c:
            for sigma in self.possible_sigma:
                model = GaussianSvmRegression(c=c, sigma=sigma)
                model.fit(x_train, y_train.flatten())
                score = model.model.score(x_val, y_val)
                if score > best_score:
                    best_score = score
                    best_model = model
        return best_model
