import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
x = np.array([-2.0, -1.0, 1.0, 2.0])
y = np.array([0.0, 1.0, 0.0, 1.0])
X = np.column_stack([np.ones(4), x])

def objective(theta):
    z = X @ theta
    return np.mean(np.logaddexp(0.0, z) - y * z)

def gradient(theta):
    return X.T @ (expit(X @ theta) - y) / len(y)
theta0 = np.zeros(2)
np.testing.assert_allclose(objective(theta0), np.log(2.0))
np.testing.assert_allclose(gradient(theta0), [0.0, -0.25])
epsilon = 1e-05
numerical = np.array([(objective(theta0 + epsilon * e) - objective(theta0 - epsilon * e)) / (2 * epsilon) for e in np.eye(2)])
np.testing.assert_allclose(gradient(theta0), numerical, atol=1e-09)
result = minimize(objective, theta0, jac=gradient, method='BFGS', options={'gtol': 1e-10})
library = LogisticRegression(C=100000000.0, solver='lbfgs', tol=1e-12, max_iter=2000).fit(x[:, None], y)
probability = expit(X @ result.x)
np.testing.assert_allclose(probability, library.predict_proba(x[:, None])[:, 1], atol=1e-06)
np.testing.assert_allclose(objective(result.x), log_loss(y, probability))
assert np.linalg.norm(gradient(result.x)) < 1e-07
extreme_z = np.array([-1000.0, 1000.0])
assert np.isfinite(np.logaddexp(0.0, extreme_z)).all()
print('initial gradient:', gradient(theta0))
print('fitted coefficients:', np.round(result.x, 8), 'loss:', round(result.fun, 8))
