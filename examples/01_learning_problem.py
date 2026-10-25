import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
X_train = np.arange(4.0).reshape(-1, 1)
y_train = np.array([1.0, 3.0, 5.0, 7.0])
X_test = np.array([[4.0], [5.0]])
y_test = np.array([9.0, 11.0])
model = LinearRegression().fit(X_train, y_train)
baseline = DummyRegressor(strategy='mean').fit(X_train, y_train)
prediction = model.predict(X_test)
mae = np.abs(prediction - y_test).mean()
baseline_mae = mean_absolute_error(y_test, baseline.predict(X_test))
np.testing.assert_allclose([model.intercept_, model.coef_[0]], [1.0, 2.0])
np.testing.assert_allclose(mae, mean_absolute_error(y_test, prediction))
np.testing.assert_allclose([mae, baseline_mae], [0.0, 6.0], atol=1e-12)
print('predictions:', prediction, 'MAE:', round(mae, 12))
print('training-mean baseline MAE:', baseline_mae)
