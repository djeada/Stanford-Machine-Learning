## Regularization

Regularization is a technique used to prevent overfitting in machine learning models, ensuring they perform well not only on the training data but also on new, unseen data.

### Overfitting in Machine Learning

- **Issue**: A model might fit the training data too closely, capturing noise rather than the underlying pattern.
- **Effect**: Poor performance on new data.
- **Manifestation in Regression**: Occurs when using higher-degree polynomials which results in a high variance hypothesis.

#### Example: Overfitting in Logistic Regression

![overfitting_logistic_regression](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/overfitting_logistic_regression.png)

### Regularization in Cost Function
Regularization works by adding a penalty term to the cost function that penalizes large coefficients, thereby reducing the complexity of the model.

#### Regularization in Linear Regression

- **Regularized Cost Function**:

$$
\min \frac{1}{2m} \left[ \sum_{i=1}^{m}(h_{\theta}(x^{(i)}) - y^{(i)})^2 + \lambda \sum_{j=1}^{m} \theta_j^2 \right]
$$

- **Penalization**: Large values of $\theta_3$ and $\theta_4$ are penalized, leading to simpler models.

![optimization_regularization](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/optimization_regularization.png)

### Regularization Parameter: $\lambda$

- **Role of $\lambda$**: Controls the trade-off between fitting the training set well and keeping the model simple (smaller parameter values).
- **Selection**: Automated methods can be used to choose an appropriate $\lambda$.

### Modifying Gradient Descent
The gradient descent algorithm can be adjusted to include the regularization term:

I. For $\theta_0$ (no regularization):

$$
\frac{\partial}{\partial \theta_0} J(\theta) = \frac{1}{m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})x_0^{(i)}
$$

II. For $\theta_j$ ($j \geq 1$):

$$
\frac{\partial}{\partial \theta_j} J(\theta) = \left( \frac{1}{m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})x_j^{(i)} \right) + \frac{\lambda}{m}\theta_j 
$$

### Regularized Linear Regression

Regularized linear regression incorporates a regularization term in the cost function and its optimization to control model complexity and prevent overfitting.

#### Gradient Descent with Regularization

To optimize the regularized linear regression model using gradient descent, the algorithm is adjusted as follows:

```
while not converged:
  for j in [0, ..., n]:
      θ_j := θ_j - α [ \frac{1}{m} \sum_{i=1}^{m}(h_{θ}(x^{(i)}) + y^{(i)})x_j^{(i)} + \frac{λ}{m} θ_j ]
```

#### Regularization with the Normal Equation

In the normal equation approach for regularized linear regression, the optimal $θ$ is computed as follows:

![regularized_normal_equation](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/regularized_normal_equation.png)

The equation includes an additional term $λI$ to the matrix being inverted, ensuring regularization is accounted for in the solution.

### Regularized Logistic Regression

The cost function for logistic regression with regularization is:

$$
J(θ) = \frac{1}{m} \sum_{i=1}^{m}[-y^{(i)}\log(h_{θ}(x^{(i)})) - (1-y^{(i)})\log(1 - h_{θ}(x^{(i)}))] + \frac{λ}{2m}\sum_{j=1}^{n}θ_j^2
$$

#### Gradient of the Cost Function

The gradient is defined for each parameter $θ_j$:

I. For $j = 0$ (no regularization on $θ_0$):

$$
\frac{\partial}{\partial θ_0} J(θ) = \frac{1}{m} \sum_{i=1}^{m} (h_{θ}(x^{(i)}) - y^{(i)})x_j^{(i)} 
$$

II. For $j ≥ 1$ (includes regularization):

$$
\frac{\partial}{\partial θ_j} J(θ) = ( \frac{1}{m} \sum_{i=1}^{m} (h_{θ}(x^{(i)}) - y^{(i)})x_j^{(i)} ) + \frac{λ}{m}θ_j 
$$

#### Optimization

For both linear and logistic regression, the gradient descent algorithm is updated to include regularization:

```
while not converged:
  for j in [0, ..., n]:
      θ_j := θ_j - α [ \frac{1}{m} \sum_{i=1}^{m}(h_{θ}(x^{(i)}) + y^{(i)})x_j^{(i)} + \frac{λ}{m} θ_j ]
```

The key difference in logistic regression lies in the hypothesis function $h_{θ}(x)$, which is based on the logistic (sigmoid) function.
