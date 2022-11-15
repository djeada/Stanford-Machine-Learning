import numpy as np

from algorithms import LogisticRegression


def accuracy(
    test_x: np.ndarray, test_y: np.ndarray, model: LogisticRegression
) -> float:
    """
    Computes the accuracy of the model on the test set.

    :param test_x: The test data.
    :param test_y: The test labels.
    :param model: The trained logistic regression model.
    :return: The accuracy of the model on the test set.
    """
    predictions = model.predict(test_x)
    return np.mean(predictions == test_y)
