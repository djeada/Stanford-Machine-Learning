import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC
X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
y = np.array([-1, -1, 1, 1])
linear = SVC(kernel='linear', C=1000000.0, tol=1e-10).fit(X, y)
np.testing.assert_allclose(linear.coef_, [[1.0]], atol=1e-08)
np.testing.assert_allclose(linear.intercept_, [0.0], atol=1e-08)
np.testing.assert_allclose(np.sort(linear.support_vectors_[:, 0]), [-1.0, 1.0])
geometric_margin = 1 / np.linalg.norm(linear.coef_)
np.testing.assert_allclose(geometric_margin, 1.0)
points = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
labels = np.array([-1, 1, 1, -1])
gamma = 2.0
rbf = SVC(kernel='rbf', gamma=gamma, C=10.0, tol=1e-10).fit(points, labels)
kernel = np.exp(-gamma * np.sum((points[:, None, :] - rbf.support_vectors_[None, :, :]) ** 2, axis=2))
manual = kernel @ rbf.dual_coef_[0] + rbf.intercept_[0]
np.testing.assert_allclose(kernel, rbf_kernel(points, rbf.support_vectors_, gamma=gamma))
np.testing.assert_allclose(manual, rbf.decision_function(points), atol=1e-10)
np.testing.assert_array_equal(rbf.predict(points), labels)
assert np.linalg.eigvalsh(rbf_kernel(points, gamma=gamma)).min() > 0.0
print('linear w/b/margin:', linear.coef_[0], linear.intercept_, geometric_margin)
print('RBF XOR decisions:', np.round(manual, 6))
