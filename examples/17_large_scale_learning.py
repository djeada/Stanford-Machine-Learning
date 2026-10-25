import numpy as np
from sklearn.linear_model import LinearRegression, SGDRegressor
x = np.linspace(-1.0, 1.0, 6)
X = np.column_stack([np.ones(6), x])
y = 1 + 2 * x
theta = np.zeros(2)

def gradient(rows):
    return X[rows].T @ (X[rows] @ theta - y[rows]) / len(X[rows])
groups = [np.arange(2), np.arange(2, 6)]
full = gradient(np.arange(6))
weighted = sum((len(g) * gradient(g) for g in groups)) / len(X)
unweighted = np.mean([gradient(g) for g in groups], axis=0)
np.testing.assert_allclose(weighted, full)
assert not np.allclose(unweighted, full)
model = SGDRegressor(loss='squared_error', penalty=None, learning_rate='constant', eta0=0.02, random_state=0, shuffle=False)
for _ in range(600):
    for rows in groups:
        model.partial_fit(x[rows, None], y[rows])
reference = LinearRegression().fit(x[:, None], y)
np.testing.assert_allclose(model.predict(x[:, None]), reference.predict(x[:, None]), atol=1e-05)
print('full/weighted gradient:', np.round(full, 8), np.round(weighted, 8))
print('unweighted worker-mean gradient:', np.round(unweighted, 8))
print('streamed SGD intercept/slope:', np.round(model.intercept_, 6), np.round(model.coef_, 6))
