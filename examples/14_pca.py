import numpy as np
from sklearn.decomposition import PCA
X = np.array([[8.0, 19.0], [8.0, 21.0], [12.0, 19.0], [12.0, 21.0]])
mean = X.mean(axis=0)
centered = X - mean
U, singular_values, Vt = np.linalg.svd(centered, full_matrices=False)
basis = Vt[:1].T
scores = centered @ basis
reconstruction = scores @ basis.T + mean
library = PCA(n_components=1, svd_solver='full').fit(X)
np.testing.assert_allclose(basis @ basis.T, library.components_.T @ library.components_)
np.testing.assert_allclose(reconstruction, library.inverse_transform(library.transform(X)))
population_eigenvalues = singular_values ** 2 / len(X)
sample_eigenvalues = singular_values ** 2 / (len(X) - 1)
np.testing.assert_allclose(population_eigenvalues, [4.0, 1.0])
np.testing.assert_allclose(library.explained_variance_, sample_eigenvalues[:1])
np.testing.assert_allclose(library.explained_variance_ratio_, [0.8])
error = np.sum((X - reconstruction) ** 2) / len(X)
np.testing.assert_allclose(error, population_eigenvalues[1:].sum())
np.testing.assert_allclose(error, 1.0)
print('mean:', mean, '1/m covariance eigenvalues:', population_eigenvalues)
print('retained variance:', library.explained_variance_ratio_[0], 'mean squared reconstruction distance:', error)
