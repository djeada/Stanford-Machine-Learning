import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
glyphs = {'I': np.array([[1, 1, 1], [0, 1, 0], [1, 1, 1.0]]), 'L': np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1.0]]), 'T': np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0.0]])}
rng = np.random.default_rng(0)
training = []
labels = []
for label, glyph in glyphs.items():
    for _ in range(30):
        training.append(np.clip(glyph + rng.normal(0, 0.08, glyph.shape), 0, 1).ravel())
        labels.append(label)
recognizer = KNeighborsClassifier(n_neighbors=1).fit(training, labels)
strip = np.concatenate([part for c in 'ILT' for part in [glyphs[c], np.zeros((3, 1))]], axis=1)
active = (strip > 0.5).any(axis=0)
edges = np.diff(np.r_[False, active, False].astype(int))
starts, ends = (np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
crops = [strip[:, left:right].ravel() for left, right in zip(starts, ends)]
recognized = ''.join(recognizer.predict(crops))
assert recognized == 'ILT'
np.testing.assert_array_equal(starts, [0, 4, 8])
np.testing.assert_array_equal(ends, [3, 7, 11])
print('segmented half-open column intervals:', [(int(a), int(b)) for a, b in zip(starts, ends)])
print('recognized synthetic text:', recognized)
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, stratify=digits.target, random_state=0)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
model.fit(X_train, y_train)
prediction = model.predict(X_test)
accuracy = accuracy_score(y_test, prediction)
assert prediction.shape == y_test.shape and accuracy > 0.85
np.testing.assert_allclose(model.named_steps['standardscaler'].mean_, X_train.mean(axis=0))
print('held-out digit accuracy:', round(accuracy, 6), 'test images:', len(y_test))
