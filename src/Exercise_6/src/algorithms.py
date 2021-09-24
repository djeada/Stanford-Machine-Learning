""""
The goal of this module is to implement all algorithms and numerical
methods needed to solve the Task 6 from the coding homeworks in the
Machine Learning course on coursera.com.
"""
from typing import Any

import numpy as np
from sklearn import svm


def train_svm(x: np.ndarray, y: np.ndarray, **params: dict[str, Any]) -> svm.SVC:
    svm_function = svm.SVC(**params)
    return svm_function.fit(x, y.flatten())


def gaussian_kernel(x1: np.ndarray, x2: np.ndarray, sigma: int) -> np.ndarray:
    sigma_squared = np.power(sigma, 2)
    return np.exp(-(x1 - x2).T.dot(x1 - x2) / (2 * sigma_squared))


def extract_features(tokens: list, num_words: int) -> np.ndarray:
    features = np.zeros(num_words)
    features[tokens] = 1
    return features
