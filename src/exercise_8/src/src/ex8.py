""""
The goal of this module is to give a comprehensive solution to Task 8
from the coding homeworks from the Machine Learning course on coursera.com.
The task is broken down into smaller parts.
"""

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

import readers
import algorithms
import plots

DATA_PATH_1 = Path("../data/data1.mat")
DATA_PATH_2 = Path("../data/data2.mat")
DATA_PATH_3 = Path("../data/movies.mat")
DATA_PATH_4 = Path("../data/movieParams.mat")
DATA_PATH_5 = Path("../data/movie_ids.txt")


def part_1() -> None:
    """
    The goal of this function is to plot the data from the first exercise.
    Returns:
      None
    """
    x, x_val, y_val = readers.read_data(DATA_PATH_1)
    plots.plot_data((x[:, 0], x[:, 1]))

    plots.plot_data((x[:, 0], x[:, 1]), "gaussian contours")
    plots.plot_gaussian_contours((x, 0, 35, 0, 35))

    mu, sigma_2 = algorithms.get_gaussian_parameters(x)
    z = algorithms.compute_gauss(x_val, mu, sigma_2)
    best_f1, best_eps = algorithms.select_threshold(y_val, z)

    # The expected value of epsilon is  8.99e-05.
    print(best_f1, best_eps)

    plots.plot_data((x[:, 0], x[:, 1]), "anomalies")
    plots.plot_gaussian_contours((x, 0, 35, 0, 35))
    plots.plot_anomalies(x, best_eps)


def part_2() -> None:
    """

    Returns:
      None
    """
    x, x_val, y_val = readers.read_data(DATA_PATH_2)

    mu, sigma_2 = algorithms.get_gaussian_parameters(x)
    z = algorithms.compute_gauss(x_val, mu, sigma_2)
    best_f1, best_eps = algorithms.select_threshold(y_val, z)
    gauss_values = algorithms.compute_gauss(x, mu, sigma_2)
    anomalies = np.array(
        [x[i] for i in range(x.shape[0]) if gauss_values[i] < best_eps]
    )

    # The expected value of epsilon is 1.38e-18, and there should be approx. 117 anomalies.
    print(best_f1, best_eps)
    print(f"number of anomalies found: {len(anomalies)}")


def part_3() -> None:
    """

    Returns:
      None
    """
    y, r = readers.read_movie_data(DATA_PATH_3)
    print(f"Average rating for movie 1: {np.mean(y[0, r[0, :]])}")
    plots.plot_movies_data(y)

    num_users = 4
    num_movies = 5
    num_features = 3
    x, theta = readers.read_movie_params(DATA_PATH_4)
    x = x[:num_movies, :num_features]
    theta = theta[:num_users, :num_features]
    y = y[:num_movies, :num_users]
    r = r[:num_movies, :num_users]
    cost = algorithms.collaborative_filtering_cost(x, theta, y, r, 0)
    print(cost)

    cost = algorithms.collaborative_filtering_cost(x, theta, y, r, 1.5)
    print(cost)


def part_4() -> None:
    """

    Returns:
      None
    """
    y, r = readers.read_movie_data(DATA_PATH_3)
    new_ratings, y, r = algorithms.update_matrices_with_new_ratings(y, r)
    movies = readers.read_movie_ids(DATA_PATH_5)

    num_users = 4
    num_movies = 5
    num_features = 3
    y = y[:num_movies, :num_users]
    r = r[:num_movies, :num_users]

    result = algorithms.optimize_theta(y, r, _lambda=10)
    x = result[: num_movies * num_features].reshape((num_movies, num_features))
    theta = result[num_movies * num_features :].reshape((num_users, num_features))
    y_norm, y_mean = algorithms.normalize_ratings(y, r)

    prediction_matrix = x.dot(theta.T)
    predictions = prediction_matrix[:, -1] + y_mean.flatten()
    indices = np.argsort(predictions)[::-1]

    print("Top recommendations:")
    for i in indices:
        print(
            f"The movie {movies[i]} was assigned the following rating: {predictions[i]:.1f}"
        )

    print("\nOriginal ratings:")
    for i, rating in enumerate(new_ratings):
        if rating > 0:
            print(f"The original rating for the movie {movies[i]} was: {rating[0]}")


def main() -> None:
    """
    The main functions. Calls the functions that implement different parts of the solution
    to the Task 8 from the coding homeworks from the Machine Learning course on coursera.com.

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
