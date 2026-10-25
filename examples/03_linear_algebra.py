import numpy as np
X = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
y = np.array([1.0, 2.0, 2.0])
theta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
residual = y - X @ theta
np.testing.assert_allclose(X.T @ residual, 0.0, atol=1e-12)
assert X.shape == (3, 2) and (X @ theta).shape == (3,) and (rank == 2)
duplicated = np.column_stack([X, X[:, 1]])
minimum_norm, _, rank_dup, _ = np.linalg.lstsq(duplicated, y, rcond=None)
alternative = minimum_norm + np.array([0.0, 10.0, -10.0])
assert rank_dup == 2 < duplicated.shape[1]
np.testing.assert_allclose(duplicated @ minimum_norm, duplicated @ alternative)
np.testing.assert_allclose(duplicated @ minimum_norm, X @ theta)
np.testing.assert_allclose(minimum_norm, [2 / 3, 1 / 4, 1 / 4])
print('residual:', np.round(residual, 8))
print('rank:', rank_dup, 'minimum-norm coefficients:', np.round(minimum_norm, 8))
