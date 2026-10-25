import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
x = np.array([1.0, 2.0, 3.0])
y = np.array([1.0, 2.0, 2.0])
X = np.column_stack([np.ones(3), x])

def cost(theta):
    return np.mean((X @ theta - y) ** 2) / 2

def gradient(theta):
    return X.T @ (X @ theta - y) / len(y)
theta = np.zeros(2)
g0 = gradient(theta)
first = theta - 0.1 * g0
np.testing.assert_allclose(g0, [-5 / 3, -11 / 3])
assert cost(first) < cost(theta)
for _ in range(3000):
    theta -= 0.1 * gradient(theta)
least_squares = np.linalg.lstsq(X, y, rcond=None)[0]
library = LinearRegression().fit(x[:, None], y)
np.testing.assert_allclose(theta, [2 / 3, 1 / 2], atol=1e-08)
np.testing.assert_allclose(theta, least_squares, atol=1e-08)
np.testing.assert_allclose(X @ theta, library.predict(x[:, None]), atol=1e-08)
np.testing.assert_allclose(cost(theta), 1 / 36)
np.testing.assert_allclose(2 * cost(theta), mean_squared_error(y, X @ theta))
print('first gradient:', g0, 'first update:', first)
print('coefficients:', np.round(theta, 8), 'half-MSE:', round(cost(theta), 8))
