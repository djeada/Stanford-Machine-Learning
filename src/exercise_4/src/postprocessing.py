import numpy as np

from algorithms import ModelBase


def accuracy(test_x: np.ndarray, test_y: np.ndarray, model: ModelBase) -> float:
    """
    Computes the accuracy of the model on the test set.

    :param test_x: The test data.
    :param test_y: The test labels.
    :param model: The trained logistic regression model.
    :return: The accuracy of the model on the test set.
    """

    # check how often model.predict(row) == test_y[row]
    # return the percentage of correct predictions
    correct = 0
    for i in range(len(test_x)):
        if model.predict(test_x[i]) == test_y[i]:
            correct += 1
    return correct / len(test_x)
