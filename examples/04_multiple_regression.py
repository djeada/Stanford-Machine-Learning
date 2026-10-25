import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
X = np.array([[1.0, 10.0], [2.0, 10.0], [1.0, 20.0], [2.0, 20.0]])
y = np.array([10.0, 12.0, 15.0, 17.0])
mean, scale = (X.mean(axis=0), X.std(axis=0))
Z = (X - mean) / scale
design = np.column_stack([np.ones(len(X)), Z])
theta = np.linalg.lstsq(design, y, rcond=None)[0]
raw_slopes = theta[1:] / scale
raw_intercept = theta[0] - mean @ raw_slopes
pipeline = make_pipeline(StandardScaler(), LinearRegression()).fit(X, y)
new = np.array([[3.0, 30.0]])
manual = theta[0] + (new - mean) / scale @ theta[1:]
np.testing.assert_allclose(theta, [13.5, 1.0, 2.5])
np.testing.assert_allclose([raw_intercept, *raw_slopes], [3.0, 2.0, 0.5])
np.testing.assert_allclose(pipeline.predict(new), manual)
np.testing.assert_allclose(manual, [24.0])
np.testing.assert_allclose(pipeline.named_steps['standardscaler'].mean_, mean)
print('scaled coefficients:', theta)
print('raw coefficients:', np.r_[raw_intercept, raw_slopes], 'new prediction:', manual)
