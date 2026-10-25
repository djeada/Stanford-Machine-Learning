import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score
from sklearn.mixture import GaussianMixture
training = np.array([[0.5, 1.2], [0.6, 1.4], [0.8, 1.3], [0.7, 1.5], [0.9, 1.7], [0.6, 1.3]])
mean = training.mean(axis=0)
centered = training - mean
covariance = centered.T @ centered / len(training)

def log_density(X):
    delta = np.atleast_2d(X) - mean
    solved = np.linalg.solve(covariance, delta.T).T
    sign, logdet = np.linalg.slogdet(covariance)
    assert sign > 0
    return -0.5 * (len(mean) * np.log(2 * np.pi) + logdet + np.sum(delta * solved, axis=1))
validation = np.array([[0.7, 1.4], [0.6, 1.3], [1.5, 0.4], [1.7, 0.3]])
truth = np.array([0, 0, 1, 1])
manual = log_density(validation)
np.testing.assert_allclose(manual, multivariate_normal.logpdf(validation, mean=mean, cov=covariance))
library = GaussianMixture(n_components=1, covariance_type='full', reg_covar=0.0, random_state=0, tol=1e-10).fit(training)
np.testing.assert_allclose(manual, library.score_samples(validation), atol=1e-09)
anomaly_score = -manual
candidates = np.r_[np.nextafter(anomaly_score.min(), -np.inf), np.unique(anomaly_score)]
threshold = max(candidates, key=lambda t: f1_score(truth, anomaly_score > t))
np.testing.assert_array_equal(anomaly_score > threshold, truth)
scaled_log = multivariate_normal.logpdf(validation * 100, mean=mean * 100, cov=covariance * 10000)
np.testing.assert_allclose(scaled_log, manual - 2 * np.log(100))
print('Gaussian mean:', np.round(mean, 8), 'MLE covariance:\n', np.round(covariance, 8))
print('validation log densities:', np.round(manual, 6))
print('validation-selected negative-log-density threshold:', round(threshold, 6))
