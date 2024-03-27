## Linear Regression with Multiple Variables
Multiple linear regression extends the concept of simple linear regression to multiple independent variables. This technique models a dependent variable as a linear combination of several independent variables.

- **Variables**: The model uses several variables (or features), such as house size, number of bedrooms, floors, and age.
- $n$: Number of features (e.g., 4 in our example).
- $m$: Number of examples (rows in data).
- $x^i$: Feature vector of the $i^{th}$ training example.
- $x_j^i$: Value of the $j^{th}$ feature in the $i^{th}$ training set.

### Hypothesis Function
The hypothesis in multiple linear regression combines all the features:

$$ h_{\theta}(x) = \theta_0 + \theta_1x_1 + \theta_2x_2 + \theta_3x_3 + \theta_4x_4 $$

For simplification, introduce $x_0 = 1$ (bias term), making the feature vector $n + 1$-dimensional.

$$ h_{\theta}(x) = \theta^T X $$

### Cost Function
The cost function measures the discrepancy between the model's predictions and actual values. It is defined as:

$$ J(\theta_0, \theta_1, ..., \theta_n) = \frac{1}{2m} \sum_{i=1}^{m}(h_{\theta}(x^{(i)}) - y^{(i)})^2 $$

### Gradient Descent for Multiple Variables
Gradient descent is employed to find the optimal parameters that minimize the cost function.

```plaintext
θ = [0] * n
while not converged:
  for j in [0, ..., n]:
      θ_j := θ_j - α ∂/∂θ_j J(θ_0, ..., θ_n)
```

- **Simultaneous Update**: Adjust each $\theta_j$​ (from 0 to nn) simultaneously.
- **Learning Rate ($\alpha$)**: Determines the step size in each iteration.
- **Partial Derivative**: Represents the direction and rate of change of the cost function with respect to each $\theta_j$​.

### Feature Scaling

When features have different scales, gradient descent may converge slowly.

- **Example**: If $x_1$ = size (0 - 2000 feet) and $x_2$ = number of bedrooms (1-5), the cost function contours become skewed, leading to a longer path to convergence.
- **Solution**: Scale the features to have comparable ranges.

![feature_scaling](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/feature_scaling.png)

### Mean Normalization

Adjust each feature $x_i$ by subtracting the mean and dividing by the range (max - min).

![mean_normalization](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/mean_normalization.png)

Transforms the features to have approximately zero mean, aiding in faster convergence.

### Learning Rate $\alpha$

- **Monitoring**: Plot $\min J(\theta)$ against the number of iterations to observe how $J(\theta)$ decreases.
- **Signs of Proper Functioning**: $J(\theta)$ should decrease with every iteration.
- **Iterative Adjustment**: Avoid hard-coding iteration thresholds. Instead, use results to adjust future runs.

![min_cost_function](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/min_cost_function.png)

### Automatic Convergence Tests

- **Goal**: Determine if $J(\theta)$ changes by a small enough threshold, indicating convergence.
- **Challenge**: Choosing an appropriate threshold can be difficult.
- **Indicator of Incorrect $\alpha$**: If $J(\theta)$ increases, it suggests the need for a smaller learning rate.

![alpha_big](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/alpha_big.png)

- **Adjusting $\alpha$**: If the learning rate is too high, you might overshoot the minimum. Conversely, if $\alpha$ is too small, convergence becomes inefficiently slow.

![alpha_small](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/alpha_small.png)

### Features and Polynomial Regression

- **Polynomial Regression**: An advanced form of linear regression where the relationship between the independent variable $x$ and the dependent variable $y$ is modeled as an $n^{th}$ degree polynomial.
- **Application**: Can fit data better than simple linear regression by capturing non-linear relationships.
- **Consideration**: Balancing between overfitting (too complex model) and underfitting (too simple model).

![polynomial_regression](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/polynomial_regression.png)

### Normal Equation

- **Normal Equation**: Provides an analytical solution to the linear regression problem.
- **Alternative to Gradient Descent**: Unlike the iterative nature of gradient descent, the normal equation solves for $\theta$ directly.

#### Procedure

- $\theta$ is an $n+1$ dimensional vector.
- The cost function $J(\theta)$ takes $\theta$ as an argument.
- Minimize $J(\theta)$ by setting its partial derivatives with respect to $\theta_j$ to zero and solving for each $\theta_j$.

#### Example

- Given $m=4$ training examples and $n=4$ features.
- Add an extra column $x_0$ (bias term) to form an $[m \times n+1]$ matrix (design matrix X).
- Construct a $[m \times 1]$ column vector $y$.
- Calculate $\theta$ using:

$$ \theta = (X^TX)^{-1}X^Ty $$

![normal_eq_table](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/normal_eq_table.png)
![normal_eq_matrix](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/normal_eq_matrix.png)

The computed $\theta$ values minimize the cost function for the given training data.

### Gradient Descent vs Normal Equation

Comparing these two methods helps understand their practical applications:

| Aspect | Gradient Descent | Normal Equation |
| ------ | ---------------- | --------------- |
| Learning Rate | Requires selecting a learning rate | No learning rate needed |
| Iterations | Numerous iterations needed | Direct computation without iterations |
| Efficiency | Works well for large $n$ (even millions) | Becomes slow for large $n$ |
| Use Case | Preferred for very large feature sets | Ideal for smaller feature sets |

Understanding when to use polynomial regression, and choosing between gradient descent and the normal equation, is crucial in developing efficient and effective linear regression models.
