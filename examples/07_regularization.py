import numpy as np
from sklearn.linear_model import Ridge
x = np.array([-1.0, 0.0, 1.0])
y = np.array([1.0, 3.0, 5.0])
X = np.column_stack([np.ones(3), x])
lam = 2.0
penalty = np.diag([0.0, 1.0])
theta = np.linalg.solve(X.T @ X + lam * penalty, X.T @ y)
library = Ridge(alpha=lam).fit(x[:, None], y)
residual = X @ theta - y
gradient = (X.T @ residual + lam * penalty @ theta) / len(y)
course_cost = (residual @ residual + lam * theta[1] ** 2) / (2 * len(y))
np.testing.assert_allclose(theta, [3.0, 1.0])
np.testing.assert_allclose(gradient, 0.0, atol=1e-12)
np.testing.assert_allclose(X @ theta, library.predict(x[:, None]))
np.testing.assert_allclose(course_cost, 2 / 3)
print('ridge coefficients:', theta, 'training SSE:', residual @ residual)
print('penalized course objective:', round(course_cost, 8))
