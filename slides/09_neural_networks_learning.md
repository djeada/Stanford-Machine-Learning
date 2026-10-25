# Neural Networks: Learning and Classification

Learning minimizes an explicit loss over network parameters. Forward propagation calculates predictions; backpropagation applies the chain rule to calculate derivatives of that loss. An optimizer then uses those derivatives to update weights and biases. Keeping these operations separate makes their shapes, objective conventions, and implementation checks easier to follow.

## Classification Problems in Neural Networks

- **Training Set Representation:** Typically represented as $\{(x^1, y^1), (x^2, y^2), ..., (x^n, y^n)\}$, where $x^i$ is the $i^{th}$ input and $y^i$ is the corresponding target output.
- $L$: Total number of layers in the network.
- $s_l$: Number of units (excluding the bias unit) in layer $l$.

### Example Network Architecture

![Example of Multi-Layer Neural Network](https://user-images.githubusercontent.com/37275728/201518449-ec13fac4-0716-4131-8e5e-f0405ce075a5.png)

In the illustrated network:

- There are 4 layers ($L=4$).
- The first layer ($s_1$) has 3 units.
- The second and third layers ($s_2$, $s_3$) each have 5 units.
- The fourth layer ($s_4$) has 4 units.

## Binary Classification

In binary classification, the neural network predicts a single output which can take one of two possible values (often represented as 0 or 1). Characteristics of binary classification in neural networks include:

- **Single Output Node:** The final layer produces a real number value, interpreted as a probability or classification decision.
- **Output Representation:** $k = 1$ (one output class) and $s_L = 1$ (one unit in the output layer).

## Multi-class Classification

Multi-class classification extends the network's capability to differentiate between multiple classes. Here, the target output $y$ is a vector representing the probability of each class. Characteristics include:

- **Multiple Output Classes:** $k$ distinct classifications are represented.
- **Output Vector:** The output $y$ is a $k$-dimensional vector of real numbers, where each element corresponds to a class.
- **Layer Configuration:** The output layer has $s_L = k$ units, each corresponding to one of the $k$ classes.

## Learning Process

The learning process in neural networks involves:

1. **Cost Function:** A generalization of the logistic regression cost function, which sums the error over all output units and classes. This function quantifies the difference between predicted and actual values.
2. **Gradient Computation:** Employing the chain rule to calculate the partial derivatives of the cost function with respect to each network parameter.
3. **Backpropagation Algorithm:** Used for efficiently computing these partial derivatives. The algorithm propagates the error signal backwards through the network, adjusting the parameters (weights and biases) to minimize the cost function.

## Neural Network Cost Function

The cost function for neural networks is an extension of the logistic regression cost function, adapted for multiple outputs. This function evaluates the performance of a neural network by measuring the difference between the predicted outputs and the actual values across all output units.

The cost function for a neural network with multiple outputs (where $K$ is the number of output units) is defined as:

$$J(\Theta) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K}[ y_k^{(i)} \log{(h_{\Theta}(x^{(i)}))}_k + (1- y_k^{(i)}) \log(1 - {(h_{\Theta} (x^{(i)}))}_k)] +  \frac{\lambda}{2m} \sum_{l=1}^{L-1} \sum_{i=1}^{s_l} \sum_{j=1}^{s_{l+1}} (\Theta^{(l)}_{ji})^2$$

- **$m$:** Number of training examples.
- **$K$:** Number of output units.
- **$L$:** Total number of layers in the network.
- **$s_l$:** Number of units in layer $l$.
- **$y_k^{(i)}$:** Actual value of the $k^{th}$ output unit for the $i^{th}$ training example.
- **$h_{\Theta}(x^{(i)})$:** Hypothesis function applied to the $i^{th}$ training example.
- **$\lambda$:** Regularization parameter to avoid overfitting.
- **$\Theta^{(l)}_{ji}$:** Weight from unit $i$ in layer $l$ to unit $j$ in layer $l+1$.

This function incorporates a sum over the $K$ output units for each training example, and also includes a regularization term to penalize large weights and prevent overfitting.

## Partial Derivative Terms

In order to optimize the cost function, we need to understand how changes in the weights $\Theta$ affect the cost. This is where the partial derivative terms come into play.

- **$\Theta^{(1)}_{ji}$:** Represents the weight from the $i^{th}$ unit in layer 1 to the $j^{th}$ unit in layer 2.
- **$\Theta^{(1)}_{10}$ , $\Theta^{(1)}_{11}$ , $\Theta^{(1)}_{21}$:** Specific weights that map from the input layer to the first and second units in the second layer, including the bias unit.

## Gradient Computation

Gradient computation in neural networks is achieved through a process known as backpropagation, which calculates the gradient of the cost function with respect to each weight in the network.

### Forward Propagation Example

Consider a neural network with one training example $(x,y)$:

- **Layer 1 (Input Layer):** $a^{(1)} = x$ and $z^{(2)} = \Theta^{(1)}a^{(1)}$.
- **Layer 2 (Hidden Layer):** $a^{(2)} = g(z^{(2)})$, append $a^{(2)}_0$, and then $z^{(3)} = \Theta^{(2)}a^{(2)}$.
- **Layer 3 (Another Hidden Layer):** $a^{(3)} = g(z^{(3)})$, append $a^{(3)}_0$, and then $z^{(4)} = \Theta^{(3)}a^{(3)}$.
- **Output Layer:** $a^{(4)} = h_{\Theta}(x) = g(z^{(4)})$.

![Gradient Computation in Neural Networks](https://user-images.githubusercontent.com/37275728/201518441-7740e76d-9a6b-426f-98ad-85a5ff207a89.png)

Each layer's output ($a^{(l)}$) becomes the input for the next layer, with the activation function $g$ (typically a sigmoid or ReLU function) applied to the weighted sums. The final output $a^{(4)}$ is the hypothesis of the network for the given input $x$. The backpropagation algorithm then computes the gradient by propagating the error backwards from the output layer to the input layer, adjusting the weights $\Theta$ to minimize the cost function $J(\Theta)$.

## Backpropagation Algorithm

Backpropagation is a key algorithm in training neural networks, used for calculating the gradient of the cost function with respect to each parameter in the network. It involves propagating errors backward through the network, from the output layer to the input layer.

### Calculating Error Terms (Deltas)

- **$\delta^{(l)}$:** The derivative of loss with respect to preactivations in layer l. It has one entry per non-bias unit in that layer, so its size is $s_l$, rather than the number of layers.
- **Last Layer Error ($\delta^{(L)}$):** For the last layer, the error is calculated as the difference between the network's output ($a^{(L)}$) and the actual value ($y$):

$$\delta^{(L)}_j = a^{(L)}_j - y_j$$

- **Error for Other Layers:** The error terms for the other layers are computed recursively using the errors from the subsequent layer:

$$\delta^{(l)}_j = \left(\sum_{k=1}^{s_{l+1}}\Theta^{(l)}_{kj}\delta^{(l+1)}_k\right)g'(z^{(l)}_j)$$

### Accumulating Gradient ($\Delta$)

- **Partial Derivative Accumulation:** $\Delta^{(l)}_{ij}$ accumulates the partial derivatives of the cost function with respect to the weights $\Theta^{(l)}_{ij}$ :

$$\Delta^{(l)}_{ij} = \Delta^{(l)}_{ij} + a^{(l)}_j \delta^{(l+1)}_i$$

- **Layer Notation:** $l$ denotes the layer, $j$ the node in layer $l$, and $i$ the error of the affected node in the subsequent layer $l+1$.

### Calculating the Gradient

The gradient of the cost function $J(\Theta)$ with respect to the weights is given by:

$$
\frac{\partial}{\partial \Theta^{(l)}_{ij}}J(\Theta) = \begin{cases}
        \frac{1}{m} \Delta^{(l)}_{ij} + \frac{\lambda}{m} \Theta ^{(l)}_{ij} \quad &\text{if} \, j \neq 0 \\
        \frac{1}{m} \Delta^{(l)}_{ij} \quad &\text{if} \, j=0 \\
   \end{cases}
$$

Here's a Python implementation of the backpropagation algorithm using mock data. We will use NumPy to handle matrix operations and simulate the forward and backward passes through a neural network.

```python
import numpy as np

# Mock data and network parameters
np.random.seed(42)
input_size = 3    # Number of input features
hidden_size = 5   # Number of hidden units
output_size = 2   # Number of output units
m = 10            # Number of training examples
lambda_reg = 0.01 # Regularization parameter

# Generate some random mock data
X = np.random.rand(m, input_size)
y = np.random.rand(m, output_size)

# Initialize weights randomly
Theta1 = np.random.rand(hidden_size, input_size + 1)
Theta2 = np.random.rand(output_size, hidden_size + 1)

# Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Derivative of sigmoid
def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Add bias unit to input layer
X_bias = np.c_[np.ones((m, 1)), X]

# Forward pass
z2 = X_bias.dot(Theta1.T)
a2 = sigmoid(z2)
a2_bias = np.c_[np.ones((a2.shape[0], 1)), a2]
z3 = a2_bias.dot(Theta2.T)
a3 = sigmoid(z3)

# Calculate error terms (deltas)
delta3 = a3 - y
delta2 = delta3.dot(Theta2[:, 1:]) * sigmoid_gradient(z2)

# Accumulate gradients
Delta1 = np.zeros(Theta1.shape)
Delta2 = np.zeros(Theta2.shape)

for i in range(m):
    Delta1 += np.outer(delta2[i], X_bias[i])
    Delta2 += np.outer(delta3[i], a2_bias[i])

# Calculate the gradient
Theta1_grad = (1/m) * Delta1
Theta2_grad = (1/m) * Delta2

# Add regularization term
Theta1_grad[:, 1:] += (lambda_reg / m) * Theta1[:, 1:]
Theta2_grad[:, 1:] += (lambda_reg / m) * Theta2[:, 1:]

print("Gradient for Theta1:", Theta1_grad)
print("Gradient for Theta2:", Theta2_grad)
```

- Backpropagation involves propagating errors backward through the network, starting from the output layer and moving to the input layer.
- Error terms, also known as deltas, are computed for each node in each layer of the network.
- For the last layer, the error is calculated as the difference between the network's output and the actual target value.
- Errors for other layers are computed recursively using the errors from the subsequent layer and the derivative of the activation function.
- The process of accumulating gradients involves calculating partial derivatives of the cost function with respect to the network's weights.
- These partial derivatives are accumulated over all training examples to compute the total gradient.
- The gradient of the cost function with respect to the weights is adjusted by averaging over the training examples and incorporating a regularization term to prevent overfitting.
- Regularization helps improve the generalization of the neural network by penalizing large weights.

The seeded gradient example prints both gradient matrices. Its soft targets exercise independent sigmoid-output derivatives; mutually exclusive class labels would instead be encoded explicitly. The following values are generated by running that example:

```text
Gradient for Theta1: [[0.0136533  0.00676217 0.00569113 0.00647519]
 [0.025253   0.0127096  0.01243469 0.01228397]
 [0.11428676 0.05134049 0.04121622 0.05063224]
 [0.11572541 0.05346377 0.04723076 0.05106483]
 [0.12357188 0.0557378  0.05115498 0.05245192]]
Gradient for Theta2: [[0.5365768  0.47289456 0.40409263 0.32762688 0.42385821 0.3859348 ]
 [0.38571614 0.34694839 0.29523439 0.24210267 0.30991434 0.2842118 ]]
```

## Gradient Checking

Due to the complexity of backpropagation, implementing it correctly is crucial. Gradient checking is a technique to validate the correctness of the computed gradients.

Steps in Gradient Checking:

1. **Function $J(\Theta)$:** Start with a function representing the cost.
2. **Perturb One Coordinate:** For parameter j, evaluate $J(\theta+\epsilon e_j)$ and $J(\theta-\epsilon e_j)$, leaving every other parameter fixed.
3. **Slope Estimation:** The derivative is approximated by the slope of the straight line joining these two points.

![Gradient Checking Visualization](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/gradient_checking.png)

Gradient checking serves as a diagnostic tool to ensure that the backpropagation implementation is correct. However, it's computationally expensive and typically used only for debugging and not during regular training.

## Derive the Output and Hidden-Layer Gradients

For one sigmoid output, Bernoulli loss has derivative $p-y$ with respect to its logit. For a batch averaged over m examples, define $\delta_2=(p-y)/m$. With hidden activations A, output weights $w_2$, and output bias $b_2$,

$$
\nabla_{w_2}J=A^\top\delta_2+\lambda w_2,\qquad
\frac{\partial J}{\partial b_2}=\sum_i(\delta_2)_i.
$$

This section uses average data loss plus $\lambda/2$ times the squared weight norms. The earlier course convention has $\lambda/(2m)$, so its regularization gradient includes an extra division by m. The implementation and finite-difference objective must use one convention consistently.

For sigmoid hidden activations,

$$
\Delta_1=(\delta_2w_2^\top)\odot A\odot(1-A),\qquad
\nabla_{W_1}J=X^\top\Delta_1+\lambda W_1,\qquad
\nabla_{b_1}J=\sum_i(\Delta_1)_{i,:}.
$$

The outer product sends each example's output derivative to each hidden unit through that unit's output weight. The elementwise factors differentiate the hidden sigmoid. Bias gradients sum across examples, and weight gradients combine those derivatives with inputs. Biases are excluded from the penalty.

## A Complete Gradient Check and Learned XOR Fit

The following network has two inputs, three hidden units, and one binary output. Its 13 parameters include both sets of biases. The same function returns loss and gradient so the optimizer receives a matched pair.

```python
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = np.array([0.0, 1.0, 1.0, 0.0])
lam = 0.001

def loss_gradient(theta):
    W1, b1 = (theta[:6].reshape(2, 3), theta[6:9])
    W2, b2 = (theta[9:12], theta[12])
    hidden = expit(X @ W1 + b1)
    z = hidden @ W2 + b2
    loss = np.mean(np.logaddexp(0.0, z) - y * z)
    loss += lam / 2 * (np.sum(W1 ** 2) + W2 @ W2)
    delta2 = (expit(z) - y) / len(y)
    delta1 = delta2[:, None] * W2 * hidden * (1 - hidden)
    gradient = np.r_[(X.T @ delta1 + lam * W1).ravel(), delta1.sum(axis=0), hidden.T @ delta2 + lam * W2, delta2.sum()]
    return (loss, gradient)
rng = np.random.default_rng(2)
initial = rng.normal(0.0, 1.0, 13)
epsilon = 1e-05
numerical = np.array([(loss_gradient(initial + epsilon * e)[0] - loss_gradient(initial - epsilon * e)[0]) / (2 * epsilon) for e in np.eye(len(initial))])
analytic = loss_gradient(initial)[1]
relative_error = np.linalg.norm(analytic - numerical) / (np.linalg.norm(analytic) + np.linalg.norm(numerical))
assert relative_error < 1e-07
result = minimize(loss_gradient, initial, jac=True, method='BFGS', options={'maxiter': 2000, 'gtol': 1e-08})
W1, b1 = (result.x[:6].reshape(2, 3), result.x[6:9])
probability = expit(expit(X @ W1 + b1) @ result.x[9:12] + result.x[12])
np.testing.assert_array_equal(probability >= 0.5, y)
assert result.fun < loss_gradient(initial)[0] / 3
assert np.linalg.norm(loss_gradient(result.x)[1]) < 1e-05
print('gradient-check relative error:', f'{relative_error:.2e}')
print('initial/final objective:', round(loss_gradient(initial)[0], 6), round(result.fun, 6))
print('learned XOR probabilities:', np.round(probability, 6))
```

The relative gradient discrepancy is about $1.46\times10^{-10}$ in the tested environment. Starting from the seeded initialization, the penalized objective falls from about 0.716415 to 0.155506. Learned probabilities are approximately $(0.042945,0.957055,0.957055,0.042945)$, reproducing the four XOR labels at threshold one half.

These checks establish a working objective, gradient, and successful optimization run for the fixture. Neural-network training is nonconvex; another initialization can reach a different stationary point. Recording the initialization, objective convention, stopping tolerance, and resulting predictions makes the example reproducible without claiming a universal optimum.

## What a Finite-Difference Check Can Miss

Central differences approximate coordinate derivatives with error of order $\epsilon^2$ for a sufficiently smooth objective. Making epsilon extremely small amplifies floating-point cancellation; making it too large measures curvature over a wider interval. A value near $10^{-5}$ is useful for this float64 fixture, rather than a universal setting.

Check all coordinates of a small model, including biases and penalized weights, before sampling coordinates in a large model. Compare gradients at more than one nontrivial point, since symmetry or zero initialization can accidentally hide an indexing error. A matching gradient does not validate label encoding, train/test separation, or whether the objective represents the intended task.

Zero-initializing all hidden units gives them identical activations and gradients, preserving their symmetry. Randomly differing parameters allow different learned features. Very large sigmoid weights can saturate units and reduce their derivatives toward zero. These numerical mechanisms explain why initialization and scaling influence training even when backpropagation itself is correct.

## Reference

These notes are based on the free video lectures offered by Stanford University, led by Professor Andrew Ng. These lectures are part of the renowned Machine Learning course available on Coursera. For more information and to access the full course, visit the [Coursera course page](https://www.coursera.org/learn/machine-learning).
