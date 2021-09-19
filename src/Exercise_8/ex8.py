import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.stats as stats
from scipy import optimize

DATA_PATH_1 = "data/data1.mat"
DATA_PATH_2 = "data/data2.mat"
DATA_PATH_3 = "data/movies.mat"
DATA_PATH_4 = "data/movieParams.mat"
DATA_PATH_5 = "data/movie_ids.txt"


def read_data(path):
    """
    x is a matrix with m rows and n columns
    """

    data = sio.loadmat(path)
    x = data["X"]
    x_val = data["Xval"]
    y_val = data["yval"]
    return x, x_val, y_val


def read_movie_data(path):
    data = sio.loadmat(path)
    y = data["Y"]
    r = data["R"]

    return y, r


def read_movie_params(path):
    data = sio.loadmat(path)
    x = data["X"]
    theta = data["Theta"]

    return x, theta


def read_movie_ids(path):
    movies = None
    with open(path, encoding="ascii", errors="surrogateescape") as _file:
        movies = [" ".join(line.strip("\n").split(" ")[1:]) for line in _file]

    return movies


def compute_f1(predictions, real_values):
    """
    F1 = 2 * (p*r)/(p+r)
    where p is precision, r is recall
    precision = "of all predicted y=1, what fraction had true y=1"
    recall = "of all true y=1, what fraction predicted y=1?
    Note predictionVec and trueLabelVec should be boolean vectors.
    """

    p, r = 0, 0
    if np.sum(predictions) != 0:
        p = np.sum(
            [real_values[x] for x in range(predictions.shape[0]) if predictions[x]]
        ) / np.sum(predictions)
    if np.sum(real_values) != 0:
        r = np.sum(
            [predictions[x] for x in range(real_values.shape[0]) if real_values[x]]
        ) / np.sum(real_values)

    return 2 * p * r / (p + r) if (p + r) else 0


def get_gaussian_parameters(x):
    mu, sigma_2 = np.mean(x, axis=0), np.var(x, axis=0)
    return mu, sigma_2


def compute_gauss(grid, mu, sigma_2):
    return stats.multivariate_normal.pdf(x=grid, mean=mu, cov=sigma_2)


def select_threshold(y_cv, cv_set, nsteps=1000):
    """
    Function to select the best epsilon value from the CV set
    by looping over possible epsilon values and computing the F1
    score for each.
    """
    epsilons = np.linspace(np.min(cv_set), np.max(cv_set), nsteps)

    best_f1, best_eps = 0, 0
    real_values = (y_cv == 1).flatten()
    for epsilon in epsilons:
        predictions = cv_set < epsilon
        f1 = compute_f1(predictions, real_values)
        if f1 > best_f1:
            best_f1 = f1
            best_eps = epsilon

    return best_f1, best_eps


def collaborative_filtering_cost(x, theta, y, r, _lambda=1.0):
    """
    Collaborative filtering cost function
    """

    j = np.sum(np.sum(((x @ theta.T - y) * r) ** 2)) / 2

    # Theta regularization
    j += np.sum(np.sum(theta ** 2)) * (_lambda / 2)
    # X regularization
    j += np.sum(np.sum(x ** 2)) * (_lambda / 2)

    return j


def collaborative_filtering_gradient(x, theta, y, r, _lambda=1.0):
    """
    Collaborative filtering gradient
    """

    # compute gradient
    x_grad = ((x @ theta.T - y) * r) @ theta
    theta_grad = ((x @ theta.T - y) * r).T @ x

    # regularization
    x_grad += _lambda * x_grad
    theta_grad += _lambda * theta_grad

    return np.hstack((x_grad.flatten(), theta_grad.flatten()))


def normalize_ratings(y, r):
    """
    Preprocess data by removing the mean rating for each film (every row).
    Without this, a user who hasn't rated any movies will have a predicted
    score of 0 for every movie, whereas in reality they should have a 
    predicted score of average score of that movie.
    """

    mean = np.sum(y, axis=1) / np.sum(r, axis=1)
    mean = mean.reshape((mean.shape[0], 1))

    return y - mean, mean


def update_matrices_with_new_ratings(y, r):
    """
    Insert new ratings into the Y matrix and the corresponding row into the R matrix.
    """

    new_ratings = np.zeros((y.shape[0], 1))
    new_ratings[0] = 4
    new_ratings[97] = 2
    new_ratings[6] = 3
    new_ratings[11] = 5
    new_ratings[53] = 4
    new_ratings[63] = 5
    new_ratings[65] = 3
    new_ratings[68] = 5
    new_ratings[182] = 4
    new_ratings[225] = 5
    new_ratings[354] = 5

    y = np.hstack((y, new_ratings))
    r = np.hstack((r, new_ratings > 0))

    return new_ratings


def optimize_theta(y, r, params=None, _lambda=0):
    if params is None:
        x = np.random.rand(5, 3)
        theta = np.random.rand(4, 3)
        params = np.concatenate((x.flatten(), theta.flatten()))

    def collaborative_filtering_cost_wrapper(params):
        x = params[: 5 * 3].reshape((5, 3))
        theta = params[5 * 3 :].reshape((4, 3))
        return collaborative_filtering_cost(x, theta, y, r, _lambda)

    def collaborative_filtering_gradient_wrapper(params):
        x = params[: 5 * 3].reshape((5, 3))
        theta = params[5 * 3 :].reshape((4, 3))
        return collaborative_filtering_gradient(x, theta, y, r, _lambda)

    return optimize.fmin_cg(
        collaborative_filtering_cost_wrapper,
        x0=params,
        fprime=collaborative_filtering_gradient_wrapper,
        maxiter=400,
        full_output=True,
    )[0]


def plot_data(data, title="scatter plot of training data"):
    x, y = data
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, marker="x", c="g", s=100)
    plt.xlabel("Latency (ms)")
    plt.ylabel("Throughput (mb/s)")
    plt.title(title)
    plt.savefig(title.lower().replace(" ", "_"))


def plot_gaussian_contours(data, n=200):

    x, x_min, x_max, y_min, y_max = data

    x_range = np.array(np.linspace(x_min, x_max, n))
    y_range = np.array(np.linspace(y_min, y_max, n))
    u, v = np.meshgrid(x_range, y_range)
    grid = np.array(list(zip(u.flatten(), v.flatten())))

    mu, sigma_2 = get_gaussian_parameters(x)
    z = compute_gauss(grid, mu, sigma_2)
    z = z.reshape(u.shape)

    plt.contour(
        x_range,
        y_range,
        z,
        np.array([10.0]) ** np.arange(-21, 0, 3.0),
        cmap=plt.cm.coolwarm,
        extend="both",
    )


def plot_anomalies(x, best_eps):
    gauss_values = compute_gauss(x, *get_gaussian_parameters(x))
    anomalies = np.array(
        [x[i] for i in range(x.shape[0]) if gauss_values[i] < best_eps]
    )
    plt.scatter(anomalies[:, 0], anomalies[:, 1], s=200, color="red", marker="x")


def plot_movies_data(y, title="movies data"):

    plt.figure(figsize=(10, 6))
    plt.imshow(y, aspect="auto")
    plt.colorbar()
    plt.ylabel(f"Movies {y.shape[0]}", fontsize=20)
    plt.xlabel(f"Users {y.shape[-1]}", fontsize=20)
    plt.title(title)
    plt.savefig(title.lower().replace(" ", "_"))


def part_1():
    x, x_val, y_val = read_data(DATA_PATH_1)
    plot_data((x[:, 0], x[:, 1]))

    plot_data((x[:, 0], x[:, 1]), "gaussian contours")
    plot_gaussian_contours((x, 0, 35, 0, 35))

    mu, sigma_2 = get_gaussian_parameters(x)
    z = compute_gauss(x_val, mu, sigma_2)
    best_f1, best_eps = select_threshold(y_val, z)

    # The expected value of epsilon is  8.99e-05.
    print(best_f1, best_eps)

    plot_data((x[:, 0], x[:, 1]), "anomalies")
    plot_gaussian_contours((x, 0, 35, 0, 35))
    plot_anomalies(x, best_eps)


def part_2():
    x, x_val, y_val = read_data(DATA_PATH_2)

    mu, sigma_2 = get_gaussian_parameters(x)
    z = compute_gauss(x_val, mu, sigma_2)
    best_f1, best_eps = select_threshold(y_val, z)
    gauss_values = compute_gauss(x, mu, sigma_2)
    anomalies = np.array(
        [x[i] for i in range(x.shape[0]) if gauss_values[i] < best_eps]
    )

    # The expected value of epsilon is 1.38e-18, and there should be approx. 117 anomalies.
    print(best_f1, best_eps)
    print(f"number of anomalies found: {len(anomalies)}")


def part_3():
    y, r = read_movie_data(DATA_PATH_3)
    print(f"Average rating for movie 1: {np.mean(y[0, r[0, :]])}")
    plot_movies_data(y)

    num_users = 4
    num_movies = 5
    num_features = 3
    x, theta = read_movie_params(DATA_PATH_4)
    x = x[:num_movies, :num_features]
    theta = theta[:num_users, :num_features]
    y = y[:num_movies, :num_users]
    r = r[:num_movies, :num_users]
    cost = collaborative_filtering_cost(x, theta, y, r, 0)
    print(cost)

    cost = collaborative_filtering_cost(x, theta, y, r, 1.5)
    print(cost)


def part_4():
    y, r = read_movie_data(DATA_PATH_3)
    new_ratings = update_matrices_with_new_ratings(y, r)
    movies = read_movie_ids(DATA_PATH_5)

    num_users = 4
    num_movies = 5
    num_features = 3
    y = y[:num_movies, :num_users]
    r = r[:num_movies, :num_users]

    result = optimize_theta(y, r, _lambda=10)
    x = result[: num_movies * num_features].reshape((num_movies, num_features))
    theta = result[num_movies * num_features :].reshape((num_users, num_features))
    y_norm, y_mean = normalize_ratings(y, r)

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


def main():
    plt.style.use("seaborn")
    part_1()
    part_2()
    part_3()
    part_4()
    plt.show()


if __name__ == "__main__":
    main()
