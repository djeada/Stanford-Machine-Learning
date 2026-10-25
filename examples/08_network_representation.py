import numpy as np
from scipy.special import expit
X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
W1 = np.array([[20.0, 20.0], [20.0, 20.0]])
b1 = np.array([-10.0, -30.0])
hidden = expit(X @ W1 + b1)
xor_probability = expit(hidden @ np.array([20.0, -40.0]) - 10.0)
xnor_probability = 1.0 - xor_probability
and_probability = hidden[:, 1]
np.testing.assert_array_equal(xor_probability >= 0.5, [0, 1, 1, 0])
np.testing.assert_array_equal(xnor_probability >= 0.5, [1, 0, 0, 1])
np.testing.assert_array_equal(and_probability >= 0.5, [0, 0, 0, 1])
assert hidden.shape == (4, 2)
print('hidden activations:\n', np.round(hidden, 6))
print('XOR:', np.round(xor_probability, 6))
print('XNOR:', np.round(xnor_probability, 6))
