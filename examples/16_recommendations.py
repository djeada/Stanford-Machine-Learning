import numpy as np
ratings = np.array([[5.0, np.nan], [3.0, 4.0], [np.nan, 1.0]])
observed = np.isfinite(ratings)
X = np.array([[1.0, 0.5], [0.5, 1.0], [1.0, 1.0]])
Theta = np.array([[1.0, 2.0], [2.0, 1.0]])
lam = 0.1

def objective_gradient(X, Theta, ratings):
    prediction = X @ Theta.T
    residual = np.zeros_like(prediction)
    residual[observed] = prediction[observed] - ratings[observed]
    loss = 0.5 * np.sum(residual ** 2) + lam / 2 * (np.sum(X ** 2) + np.sum(Theta ** 2))
    return (loss, residual @ Theta + lam * X, residual.T @ X + lam * Theta)
loss, gX, gTheta = objective_gradient(X, Theta, ratings)
np.testing.assert_allclose(loss, 9.35)
np.testing.assert_allclose(gX, [[-2.9, -5.95], [-4.45, -2.9], [4.1, 2.1]])
np.testing.assert_allclose(gTheta, [[-3.15, -1.8], [1.2, 0.1]])
packed = np.r_[X.ravel(), Theta.ravel()]

def unpack_loss(v):
    return objective_gradient(v[:6].reshape(3, 2), v[6:].reshape(2, 2), ratings)[0]
epsilon = 1e-05
numerical = np.array([(unpack_loss(packed + epsilon * e) - unpack_loss(packed - epsilon * e)) / (2 * epsilon) for e in np.eye(len(packed))])
np.testing.assert_allclose(np.r_[gX.ravel(), gTheta.ravel()], numerical, atol=1e-08)
filled = ratings.copy()
filled[~observed] = 1000000000.0
np.testing.assert_allclose(objective_gradient(X, Theta, filled)[0], loss)
means = np.nanmean(ratings, axis=1)
np.testing.assert_allclose(means, [5.0, 3.5, 1.0])

def als_step(X, Theta):
    X, Theta = (X.copy(), Theta.copy())
    for i in range(len(X)):
        mask = observed[i]
        A = Theta[mask]
        X[i] = np.linalg.solve(A.T @ A + lam * np.eye(2), A.T @ ratings[i, mask])
    for j in range(len(Theta)):
        mask = observed[:, j]
        A = X[mask]
        Theta[j] = np.linalg.solve(A.T @ A + lam * np.eye(2), A.T @ ratings[mask, j])
    return (X, Theta)
history = [loss]
for _ in range(30):
    X, Theta = als_step(X, Theta)
    history.append(objective_gradient(X, Theta, ratings)[0])
assert np.all(np.diff(history) <= 1e-10)
assert history[-1] < history[0]
print('initial cost:', loss, 'item means:', means)
print('initial gradients:\n', gX, '\n', gTheta)
print('ALS final penalized objective:', round(history[-1], 8))
