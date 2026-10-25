import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = np.array([0.0, 1.0, 1.0, 0.0])
lam = 0.001

def loss_gradient(theta):
    W1, b1 = (theta[:6].reshape(2, 3), theta[6:9])
    W2, b2 = (theta[9:12], theta[12])
    hidden = expit(X @ W1 + b1)
    z = hidden @ W2 + b2
    loss = np.mean(np.logaddexp(0.0, z) - y * z)
    loss += lam / 2 * (np.sum(W1 ** 2) + W2 @ W2)
    delta2 = (expit(z) - y) / len(y)
    delta1 = delta2[:, None] * W2 * hidden * (1 - hidden)
    gradient = np.r_[(X.T @ delta1 + lam * W1).ravel(), delta1.sum(axis=0), hidden.T @ delta2 + lam * W2, delta2.sum()]
    return (loss, gradient)
rng = np.random.default_rng(2)
initial = rng.normal(0.0, 1.0, 13)
epsilon = 1e-05
numerical = np.array([(loss_gradient(initial + epsilon * e)[0] - loss_gradient(initial - epsilon * e)[0]) / (2 * epsilon) for e in np.eye(len(initial))])
analytic = loss_gradient(initial)[1]
relative_error = np.linalg.norm(analytic - numerical) / (np.linalg.norm(analytic) + np.linalg.norm(numerical))
assert relative_error < 1e-07
result = minimize(loss_gradient, initial, jac=True, method='BFGS', options={'maxiter': 2000, 'gtol': 1e-08})
W1, b1 = (result.x[:6].reshape(2, 3), result.x[6:9])
probability = expit(expit(X @ W1 + b1) @ result.x[9:12] + result.x[12])
np.testing.assert_array_equal(probability >= 0.5, y)
assert result.fun < loss_gradient(initial)[0] / 3
assert np.linalg.norm(loss_gradient(result.x)[1]) < 1e-05
print('gradient-check relative error:', f'{relative_error:.2e}')
print('initial/final objective:', round(loss_gradient(initial)[0], 6), round(result.fun, 6))
print('learned XOR probabilities:', np.round(probability, 6))
