# Regularization

Regularization adds a preference among fitted models through a penalty or constraint. It can reduce sensitivity to training noise by accepting bias in exchange for lower variability. Its predictive value depends on the data, representation, and penalty strength, so it is assessed using validation data.

## Overfitting in Machine Learning

- **Issue**: A model might fit the training data too closely, capturing noise rather than the underlying pattern.
- **Effect**: Poor performance on new data.
- **Manifestation in Regression**: Occurs when using higher-degree polynomials which results in a high variance hypothesis.

### Example: Overfitting in Logistic Regression

![overfitting_logistic_regression](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/overfitting_logistic_regression.png)

## Regularization in Cost Function
Regularization works by adding a penalty term to the cost function that penalizes large coefficients, thereby reducing the complexity of the model.

### Regularization in Linear Regression

- **Regularized Cost Function**:

$$
\min \frac{1}{2m} \left[ \sum_{i=1}^{m}(h_{\theta}(x^{(i)}) - y^{(i)})^2 + \lambda \sum_{j=1}^{n} \theta_j^2 \right]
$$

- **Penalization**: Large values of $\theta_3$ and $\theta_4$ are penalized, leading to simpler models.

![optimization_regularization](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/optimization_regularization.png)

## Regularization Parameter: $\lambda$

- **Role of $\lambda$**: Controls the trade-off between fitting the training set well and keeping the model simple (smaller parameter values).
- **Selection**: Automated methods can be used to choose an appropriate $\lambda$.

## Modifying Gradient Descent
The gradient descent algorithm can be adjusted to include the regularization term:

I. For $\theta_0$ (no regularization):

$$
\frac{\partial}{\partial \theta_0} J(\theta) = \frac{1}{m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})x_0^{(i)}
$$

II. For $\theta_j$ ($j \geq 1$):

$$
\frac{\partial}{\partial \theta_j} J(\theta) = \left( \frac{1}{m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})x_j^{(i)} \right) + \frac{\lambda}{m}\theta_j
$$

## Regularized Linear Regression

Regularized linear regression incorporates a regularization term in the cost function and its optimization to control model complexity and prevent overfitting.

### Gradient Descent with Regularization

To optimize the regularized linear regression model using gradient descent, the algorithm is adjusted as follows:

```
while not converged:
  gradient := X.T @ (h - y) / m
  gradient[1:] := gradient[1:] + (λ/m) * θ[1:]
  θ := θ - α * gradient
```

Here is the Python code to demonstrate regularization in linear regression, including the regularized cost function and gradient descent with regularization. This example uses numpy to implement the regularized linear regression model:

```python
import numpy as np

def hypothesis(X, theta):
    return np.dot(X, theta)

def compute_cost(X, y, theta, lambda_reg):
    m = len(y)
    h = hypothesis(X, theta)
    cost = (1 / (2 * m)) * (np.sum((h - y) ** 2) + lambda_reg * np.sum(theta[1:] ** 2))
    return cost

def gradient_descent(X, y, theta, alpha, lambda_reg, num_iters):
    m = len(y)
    cost_history = np.zeros(num_iters)

    for iter in range(num_iters):
        h = hypothesis(X, theta)
        theta[0] = theta[0] - alpha * (1 / m) * np.sum((h - y) * X[:, 0])
        for j in range(1, len(theta)):
            theta[j] = theta[j] - alpha * ((1 / m) * np.sum((h - y) * X[:, j]) + (lambda_reg / m) * theta[j])

        cost_history[iter] = compute_cost(X, y, theta, lambda_reg)

    return theta, cost_history

# Example usage with mock data
np.random.seed(42)
X = np.random.rand(10, 2)  # Feature matrix (10 examples, 2 features)
y = np.random.rand(10)     # Target values

# Adding a column of ones to X for the intercept term (theta_0)
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Initial parameters
theta = np.random.randn(X.shape[1])
alpha = 0.01  # Learning rate
lambda_reg = 0.1  # Regularization parameter
num_iters = 1000  # Number of iterations

# Perform gradient descent with regularization
theta, cost_history = gradient_descent(X, y, theta, alpha, lambda_reg, num_iters)

print("Optimized parameters:", theta)
print("Final cost:", cost_history[-1])
```

### Regularization with the Normal Equation

In the normal equation approach for regularized linear regression, the optimal $θ$ is computed as follows:

![regularized_normal_equation](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/regularized_normal_equation.png)

The equation includes an additional term $λI$ to the matrix being inverted, ensuring regularization is accounted for in the solution.

Here is the Python code to implement regularized linear regression using the normal equation:

```python
import numpy as np

def regularized_normal_equation(X, y, lambda_reg):
    m, n = X.shape
    I = np.eye(n)
    I[0, 0] = 0  # Do not regularize the bias term (theta_0)

    theta = np.linalg.solve(X.T @ X + lambda_reg * I, X.T @ y)
    return theta

# Example usage with mock data
np.random.seed(42)
X = np.random.rand(10, 2)  # Feature matrix (10 examples, 2 features)
y = np.random.rand(10)     # Target values

# Adding a column of ones to X for the intercept term (theta_0)
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Regularization parameter
lambda_reg = 0.1

# Compute the optimal parameters using the regularized normal equation
theta = regularized_normal_equation(X, y, lambda_reg)

print("Optimized parameters using regularized normal equation:", theta)
```

## Regularized Logistic Regression

The cost function for logistic regression with regularization is:

$$
J(θ) = \frac{1}{m} \sum_{i=1}^{m}[-y^{(i)}\log(h_{θ}(x^{(i)})) - (1-y^{(i)})\log(1 - h_{θ}(x^{(i)}))] + \frac{λ}{2m}\sum_{j=1}^{n}θ_j^2
$$

### Gradient of the Cost Function

The gradient is defined for each parameter $θ_j$:

I. For $j = 0$ (no regularization on $θ_0$):

$$
\frac{\partial}{\partial θ_0} J(θ) = \frac{1}{m} \sum_{i=1}^{m} (h_{θ}(x^{(i)}) - y^{(i)})x_0^{(i)}
$$

II. For $j ≥ 1$ (includes regularization):

$$
\frac{\partial}{\partial θ_j} J(θ) = ( \frac{1}{m} \sum_{i=1}^{m} (h_{θ}(x^{(i)}) - y^{(i)})x_j^{(i)} ) + \frac{λ}{m}θ_j
$$

### Optimization

For both linear and logistic regression, the gradient descent algorithm is updated to include regularization:

```
while not converged:
  gradient := X.T @ (h - y) / m
  gradient[1:] := gradient[1:] + (λ/m) * θ[1:]
  θ := θ - α * gradient
```

The key difference in logistic regression lies in the hypothesis function $h_{θ}(x)$, which is based on the logistic (sigmoid) function.

Here is the Python code to implement regularized logistic regression using gradient descent:

```python
import numpy as np
from scipy.special import expit  # Sigmoid function

def sigmoid(z):
    return expit(z)

def compute_cost(X, y, theta, lambda_reg):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    z = X @ theta
    cost = np.mean(np.logaddexp(0.0, z) - y*z) + (lambda_reg / (2*m))*np.sum(theta[1:]**2)
    return cost

def gradient_descent(X, y, theta, alpha, lambda_reg, num_iters):
    m = len(y)
    cost_history = np.zeros(num_iters)

    for iter in range(num_iters):
        h = sigmoid(np.dot(X, theta))
        error = h - y

        theta[0] = theta[0] - alpha * (1 / m) * np.sum(error * X[:, 0])
        for j in range(1, len(theta)):
            theta[j] = theta[j] - alpha * ((1 / m) * np.sum(error * X[:, j]) + (lambda_reg / m) * theta[j])

        cost_history[iter] = compute_cost(X, y, theta, lambda_reg)

    return theta, cost_history

# Example usage with mock data
np.random.seed(42)
X = np.random.rand(10, 2)  # Feature matrix (10 examples, 2 features)
y = np.random.randint(0, 2, 10)  # Binary target values

# Adding a column of ones to X for the intercept term (theta_0)
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Initial parameters
theta = np.random.randn(X.shape[1])
alpha = 0.01  # Learning rate
lambda_reg = 0.1  # Regularization parameter
num_iters = 1000  # Number of iterations

# Perform gradient descent with regularization
theta, cost_history = gradient_descent(X, y, theta, alpha, lambda_reg, num_iters)

print("Optimized parameters:", theta)
print("Final cost:", cost_history[-1])
```

## Match the Objective before Matching the Library Parameter

The convention here is

$$
J(b,w)=\frac1{2m}\{\|b\mathbf1+Xw-y\|^2+\lambda\|w\|^2\}.
$$

The intercept is excluded from the penalty. Multiplying the entire objective by $2m$ leaves its minimizer unchanged. Scikit-learn's Ridge objective, squared-error sum plus alpha times squared coefficient norm, therefore matches at $\mathrm{alpha}=\lambda$. A text using average loss plus an unscaled coefficient penalty gives lambda a different sample-size dependence.

For logistic average loss plus $\lambda\|w\|^2/(2m)$, the corresponding unweighted L2 inverse penalty is $C=1/\lambda$. The comparison assumes identical sample weights, feature transformation, and intercept treatment; a solver with a penalized synthetic intercept needs additional care. The library objectives are given in the [scikit-learn linear-model guide](https://scikit-learn.org/1.8/modules/linear_model.html).

## Worked Ridge Fit with an Unpenalized Intercept

Take $x=(-1,0,1)$ and $y=(1,3,5)$. Their means are zero and three, $\sum x_i^2=2$, and $\sum x_i(y_i-3)=4$. The unregularized slope is two. At $\lambda=2$,

$$
w=\frac4{2+2}=1,\qquad b=3.
$$

Predictions are 2, 3, and 4. Squared residuals sum to two, the penalty is two, and the course objective is $4/6=2/3$. The unregularized fit has zero training residuals but penalized objective $2(2)^2/6=4/3$. Improving a penalized objective can deliberately increase training MSE.

```python
import numpy as np
from sklearn.linear_model import Ridge
x = np.array([-1.0, 0.0, 1.0])
y = np.array([1.0, 3.0, 5.0])
X = np.column_stack([np.ones(3), x])
lam = 2.0
penalty = np.diag([0.0, 1.0])
theta = np.linalg.solve(X.T @ X + lam * penalty, X.T @ y)
library = Ridge(alpha=lam).fit(x[:, None], y)
residual = X @ theta - y
gradient = (X.T @ residual + lam * penalty @ theta) / len(y)
course_cost = (residual @ residual + lam * theta[1] ** 2) / (2 * len(y))
np.testing.assert_allclose(theta, [3.0, 1.0])
np.testing.assert_allclose(gradient, 0.0, atol=1e-12)
np.testing.assert_allclose(X @ theta, library.predict(x[:, None]))
np.testing.assert_allclose(course_cost, 2 / 3)
print('ridge coefficients:', theta, 'training SSE:', residual @ residual)
print('penalized course objective:', round(course_cost, 8))
```

The same penalty matrix is used in the solve and the gradient, with zero in its intercept position. The library comparison verifies both coefficients and predictions. A reproducible comparison must match the objective before treating two penalty settings as equivalent.

## Why Feature Units Matter

Re-expressing a feature from meters to centimeters divides its coefficient by 100 for identical predictions. Its squared penalty then becomes 10,000 times smaller. An L2 penalty on raw coefficients defines a geometry in parameter space that depends on feature units.

Training-set standardization makes a common penalty more comparable across features. It changes coefficient interpretation to a one-training-standard-deviation change, and does not eliminate the need to consider unusual or scientifically important features separately.

As lambda grows, ridge slopes shrink toward zero and the unpenalized intercept approaches the training-target mean. This limiting baseline can underfit. Select the strength with held-out loss, without adding the training penalty to the validation metric.

Adding a positive penalty also increases curvature along penalized near-collinear directions and stabilizes estimation. Ridge generally shrinks coefficients continuously without setting them exactly to zero. L1 penalties have a different geometry that can create sparse solutions. Neither penalty establishes causality or corrects an unrepresentative sampling mechanism.

## Reference

These notes are based on the free video lectures offered by Stanford University, led by Professor Andrew Ng. These lectures are part of the renowned Machine Learning course available on Coursera. For more information and to access the full course, visit the [Coursera course page](https://www.coursera.org/learn/machine-learning).
