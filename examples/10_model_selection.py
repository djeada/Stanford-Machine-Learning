import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
X_train = np.arange(-2.0, 3.0).reshape(-1, 1)
y_train = X_train[:, 0] ** 2
X_test = np.array([[-1.5], [-0.5], [0.5], [1.5]])
y_test = X_test[:, 0] ** 2
line = LinearRegression().fit(X_train, y_train)
np.testing.assert_allclose(line.intercept_, 2.0)
np.testing.assert_allclose(mean_squared_error(y_train, line.predict(X_train)), 2.8)
np.testing.assert_allclose(mean_squared_error(y_test, line.predict(X_test)), 1.5625)
pipeline = Pipeline([('features', PolynomialFeatures(include_bias=False)), ('scale', StandardScaler()), ('model', LinearRegression())])
search = GridSearchCV(pipeline, {'features__degree': [1, 2]}, cv=KFold(5, shuffle=True, random_state=0), scoring='neg_mean_squared_error')
search.fit(X_train, y_train)
assert search.best_params_['features__degree'] == 2
test_mse = mean_squared_error(y_test, search.predict(X_test))
assert test_mse < 1e-20
print('linear train/test MSE:', 2.8, 1.5625)
print('CV-selected degree:', search.best_params_['features__degree'])
print('quadratic test MSE:', f'{test_mse:.2e}')
