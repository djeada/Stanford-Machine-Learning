""""
The goal of this module is to implement all algorithms and numerical
methods needed to solve the Task 6 from the coding homeworks in the
Machine Learning course on coursera.com.
"""
from typing import Any

import numpy as np
from sklearn import svm


def train_svm(x: np.ndarray, y: np.ndarray, **params: dict[str, Any]) -> svm.SVC:
    """
    Train a SVM classifier. The parameters are passed as a dictionary.

    Args:
        x: The training data.
        y: The training labels.
        params: The parameters of the SVM classifier.
    
    Returns:
        The trained SVM classifier.
    """
    svm_function = svm.SVC(**params)
    return svm_function.fit(x, y.flatten())


def gaussian_kernel(x1: np.ndarray, x2: np.ndarray, sigma: int) -> np.ndarray:
    """
    Compute the Gaussian kernel between x1 and x2.

    Args:
        x1: The first vector.
        x2: The second vector.
        sigma: The sigma of the Gaussian function.
    
    Returns:
        The Gaussian kernel between x1 and x2.
    """
    sigma_squared = np.power(sigma, 2)
    return np.exp(-(x1 - x2).T.dot(x1 - x2) / (2 * sigma_squared))


def extract_features(tokens: list, num_words: int) -> np.ndarray:
    """
    Extract the features from the tokens. The features are the words that appear in the tokens.
    
    Args:
        tokens: The tokens of the text.
        num_words: The number of words to extract.

    Returns:
        The features of the tokens.
    """
    features = np.zeros(num_words)
    features[tokens] = 1
    return features
