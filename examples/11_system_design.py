import numpy as np
from scipy.special import logsumexp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
texts = ['buy deal', 'buy', 'meeting', 'meeting deal']
y = np.array([1, 1, 0, 0])
vocabulary = ['buy', 'deal', 'meeting']
vectorizer = CountVectorizer(binary=True, vocabulary=vocabulary)
X = vectorizer.fit_transform(texts).toarray()
np.testing.assert_array_equal(X, [[1, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 1]])
word_probability = np.array([[(X[y == c, j].sum() + 1) / (sum(y == c) + 2) for j in range(X.shape[1])] for c in [0, 1]])
new = vectorizer.transform(['buy deal']).toarray()[0]
log_joint = np.log(0.5) + (new * np.log(word_probability) + (1 - new) * np.log1p(-word_probability)).sum(axis=1)
manual = np.exp(log_joint - logsumexp(log_joint))
model = BernoulliNB(alpha=1.0).fit(X, y)
np.testing.assert_allclose(manual, [0.1, 0.9])
np.testing.assert_allclose(model.predict_proba(new[None, :])[0], manual)
pipeline = make_pipeline(CountVectorizer(binary=True), BernoulliNB(alpha=1.0))
pipeline.fit(texts, y)
assert 'jackpot' not in pipeline.named_steps['countvectorizer'].vocabulary_
assert pipeline.predict_proba(['jackpot']).shape == (1, 2)
truth = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 1])
scores = np.array([0.1, 0.4, 0.35, 0.8, 0.45, 0.05, 0.6, 0.9, 0.75, 0.85])
print('new-message class probabilities:', manual)
for threshold in [0.35, 0.5, 0.8]:
    predicted = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    np.testing.assert_allclose([precision, recall, f1], [precision_score(truth, predicted, zero_division=0), recall_score(truth, predicted, zero_division=0), f1_score(truth, predicted, zero_division=0)])
    print('threshold/counts/P/R/F1:', threshold, tuple(map(int, (tn, fp, fn, tp))), np.round([precision, recall, f1], 6))
candidates = np.unique(scores)
selected = max(candidates, key=lambda t: f1_score(truth, scores >= t))
np.testing.assert_allclose(selected, 0.35)
np.testing.assert_allclose(f1_score(truth, scores >= selected), 6 / 7)
