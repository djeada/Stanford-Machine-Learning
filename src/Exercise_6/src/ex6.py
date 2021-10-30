""""
The goal of this module is to give a comprehensive solution to Task 6
from the coding homeworks from the Machine Learning course on coursera.com.
The task is broken down into smaller parts.
"""

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from pathlib import Path

import readers
import algorithms
import plots

DATA_PATH_1 = Path("../data/data1.mat")
DATA_PATH_2 = Path("../data/data2.mat")
DATA_PATH_3 = Path("../data/data3.mat")
DATA_PATH_4 = Path("../data/emailSample1.txt")
DATA_PATH_5 = Path("../data/vocab.txt")
DATA_PATH_6 = Path("../data/spamTrain.mat")
DATA_PATH_7 = Path("../data/spamTest.mat")


def part_1() -> None:
    """
    Support Vector Machines:
    - Data visualization.
    - Implementing SVM with Gaussian Kernels.

    Returns:
      None
    """
    x, y = readers.read_data(DATA_PATH_1)
    pos = np.array([x[i] for i in range(x.shape[0]) if y[i] == 1])
    neg = np.array([x[i] for i in range(x.shape[0]) if y[i] == 0])
    plots.plot_data((pos, neg))

    for c_value in (1, 100):
        title = f"Decision boundary with C={c_value}"
        plots.plot_data((pos, neg), title)
        plots.plot_boundary(
            (algorithms.train_svm(x, y, C=c_value, kernel="linear"), 0, 4.5)
        )


def part_2() -> None:
    """
    Support Vector Machines:
    - Data visualization.
    - Implementing SVM with Gaussian Kernels.

    Returns:
      None
    """
    x, y = readers.read_data(DATA_PATH_2)
    pos = np.array([x[i] for i in range(x.shape[0]) if y[i] == 1])
    neg = np.array([x[i] for i in range(x.shape[0]) if y[i] == 0])
    plots.plot_data((pos, neg), title="scatter plot of training data 2")

    c_value = 1
    sigma = 0.1
    title = f"Decision boundary with C={c_value}"
    plots.plot_data((pos, neg), title)
    plots.plot_boundary_gaussian(
        (
            algorithms.train_svm(
                x, y, C=c_value, kernel="rbf", gamma=np.power(sigma, -2)
            ),
            0,
            1,
            0.4,
            1,
        )
    )


def part_3() -> None:
    """
    Support Vector Machines:
    - Data visualization.
    - Implementing SVM with Gaussian Kernels.

    Returns:
      None
    """
    x, y = readers.read_data(DATA_PATH_3)
    pos = np.array([x[i] for i in range(x.shape[0]) if y[i] == 1])
    neg = np.array([x[i] for i in range(x.shape[0]) if y[i] == 0])
    plots.plot_data((pos, neg), title="scatter plot of training data 3")

    c_value = 2
    sigma = 0.1
    title = f"Decision boundary with C={c_value}"
    plots.plot_data((pos, neg), title)
    plots.plot_boundary_gaussian(
        (
            algorithms.train_svm(
                x, y, C=c_value, kernel="rbf", gamma=np.power(sigma, -2)
            ),
            -0.5,
            0.3,
            -0.8,
            0.6,
        )
    )


def part_4() -> None:
    """
    Spam Classification:
    - Preprocessing emails.
    - Making a vocabulary list.
    - Extracting features from emails.
    - Training SVM for spam classification.

    Returns:
      None
    """
    vocabulary = readers.read_vocabulary(DATA_PATH_5)
    tokens = readers.read_tokens(DATA_PATH_4, vocabulary)
    feature_vector_len = len(vocabulary)
    features = algorithms.extract_features(tokens, feature_vector_len)
    non_zero_count = np.count_nonzero(features)

    print(f"Length of feature vector is {feature_vector_len}")
    print(f"Number of non-zero entries is {non_zero_count}")

    x, y = readers.read_data(DATA_PATH_6)
    svm_function = algorithms.train_svm(
        x,
        y,
        C=0.1,
        coef0=0.0,
        decision_function_shape="ovr",
        degree=3,
        gamma="auto",
        kernel="linear",
    )

    predictions = svm_function.predict(x)
    print(f"Training accuracy: {np.mean(predictions == y.flatten()) * 100}")

    x_test, y_test = readers.read_test_data(DATA_PATH_7)
    predictions = svm_function.predict(x_test)
    print(f"Test accuracy: {np.mean(predictions == y_test.flatten()) * 100}")

    weights = svm_function.coef_[0]
    data_frame = pd.DataFrame({"vocabulary": vocabulary, "weights": weights})
    print(data_frame.sort_values(by="weights", ascending=False).head())


def main() -> None:
    """
    The main functions. Calls the functions that implement different parts of the solution
    to the Task 6 from the coding homeworks from the Machine Learning course on coursera.com.

    Returns:
      None
    """
    plt.style.use("seaborn")
    part_1()
    part_2()
    part_3()
    part_4()
    plt.show()


if __name__ == "__main__":
    main()
