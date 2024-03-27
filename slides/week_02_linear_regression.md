# Linear Regression in Depth

Linear Regression is a fundamental type of supervised learning algorithm in statistics and machine learning. It's utilized for modeling and analyzing the relationship between a dependent variable and one or more independent variables. The goal is to predict continuous output values based on the input variables.

- **Purpose**: To predict a continuous outcome based on one or more predictor variables.
- **Model**: The output variable (y) is assumed to be a linear combination of the input variables (x), with coefficients (θ) representing weights.

### Mathematical Model
The hypothesis function in linear regression is given by:

$$
h_{\theta}(x) = \theta_0 + \theta_1x
$$

where:

- $h_{\theta}(x)$ is the predicted value,
- $\theta_0$ is the y-intercept (bias term),
- $\theta_1$ is the slope (weight for the feature x).

### Cost Function (Mean Squared Error)
The cost function in linear regression measures how far off the predictions are from the actual outcomes. It's commonly represented as:

$$
J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^{m}(h_{\theta}(x^{(i)}) - y^{(i)})^2
$$

This is the mean squared error (MSE) cost function, where:

- $m$ is the number of training examples,
- $x^{(i)}$ and $y^{(i)}$ are the input and output of the $i^{th}$ training example.

### Optimization: Gradient Descent
To find the optimal parameters ($\theta_0$ and $\theta_1$), we use gradient descent to minimize the cost function. The gradient descent algorithm iteratively adjusts the parameters to reduce the cost.

### Notation

- $m$: Number of training examples.
- $x$: Input variables/features.
- $y$: Output variable (target variable).
- $(x, y)$: Single training example.
- $(x^{i}, y^{i})$: $i^{th}$ training example.

### Training Process

1. **Input**: Training set.
2. **Algorithm**: The learning algorithm processes this data.
3. **Output**: Hypothesis function $h$ which estimates the value of $y$ for a new input $x$.

### Example: House Price Prediction
Using linear regression, we can predict house prices based on house size.

![house price table](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/house_price_table.png)

### Analyzing the Cost Function
Different values of $\theta_1$ yield different cost (J) values:

- For $\theta_1 = 1$, $J(\theta_1) = 0$ (Ideal scenario).
- For $\theta_1 = 0.5$, $J(\theta_1) = 0.58$ (Higher error).
- For $\theta_1 = 0$, $J(\theta_1) = 2.3$ (Maximum error).

Optimization aims to find the value of $\theta_1$ that minimizes $J(\theta_1)$.

![cost_function](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/cost_function.png)

### A Deeper Insight into the Cost Function - Simplified Cost Function

In linear regression, the cost function $J(\theta_0, \theta_1)$ is a critical component, involving two parameters: $\theta_0$ and $\theta_1$. Visualization of this function can be achieved through different plots.

#### 3D Surface Plot

The 3D surface plot illustrates the cost function where:
- $X$-axis represents $\theta_1$.
- $Z$-axis represents $\theta_0$.
- $Y$-axis represents $J(\theta_0, \theta_1)$.

![surface_cost_function](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/surface_cost_function.png)

#### Contour Plots

Contour plots offer a 2D perspective:

- Each color or level represents the same value of $J(\theta_0, \theta_1)$.
- The center of concentric circles indicates the minimum of the cost function.

![contour_cost_function](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/contour_cost_function.png)

## Gradient Descent Algorithm

The gradient descent algorithm iteratively adjusts $\theta_0$ and $\theta_1$ to minimize the cost function. The algorithm proceeds as follows:

```plaintext
θ = [0, 0]
while not converged:
    for j in [0, 1]:
        θ_j := θ_j - α ∂/∂θ_j J(θ_0, θ_1)
```

- Begin with initial guesses (e.g., [0,0]).
- Continuously adjust $\theta_0$ and $\theta_1$ to reduce $J(\theta_0, \theta_1)$.
- Proceed until a local minimum is reached.

![gradient_descent](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/gradient_descent.png)

### Key Elements of Gradient Descent

- **Learning Rate ($\alpha$)**: Determines the step size during each iteration.
- **Partial Derivative**: Indicates the direction to move in the parameter space.
- A negative derivative suggests a decrease in $\theta_1$ to move towards a minimum.
- Conversely, a positive derivative suggests an increase in $\theta_1$.

### Partial Derivative vs. Derivative

- **Partial Derivative**: Applied when focusing on a single variable among several.
- **Derivative**: Utilized when considering all variables.

### At a Local Minimum
At this point, the derivative equals zero, implying no further changes in $\theta_1$:

$$
\alpha \times 0 = 0 \Rightarrow \theta_1 = \theta_1 - 0
$$

![local_minimum](https://user-images.githubusercontent.com/37275728/201476896-555ad8c4-8422-428b-937f-12cdf70d75bd.png)

Through gradient descent, the optimal $\theta_0$ and $\theta_1$ values are identified, minimizing the cost function and enhancing the linear regression model's performance.

## Linear Regression with Gradient Descent

In linear regression, gradient descent is applied to minimize the squared error cost function $J(\theta_0, \theta_1)$. The process involves calculating the partial derivatives of the cost function with respect to each parameter $\theta_0$ and $\theta_1$.

### Partial Derivatives of the Cost Function
The gradient of the cost function is computed as follows:

- For the squared error cost function:

$$\frac{\partial}{\partial \theta_j} J(\theta_0, \theta_1) = \frac{\partial}{\partial \theta_j} \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})^2$$

$$= \frac{\partial}{\partial \theta_j} \frac{1}{2m} \sum_{i=1}^{m} (\theta_0 + \theta_1x^{(i)} - y^{(i)})^2$$

The partial derivatives for each $\theta_j$ are:

- For $j=0$:

$$\frac{\partial}{\partial \theta_0} J(\theta_0, \theta_1)=\frac{\partial}{\partial \theta_0} \frac{1}{m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})$$

- For $j=1$:
  
$$\frac{\partial}{\partial \theta_1} J(\theta_0, \theta_1)=\frac{\partial}{\partial \theta_1} \frac{1}{m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})x^{(i)}$$

## Normal Equations Method

For solving the minimization problem $\min J(\theta_0, \theta_1)$, the normal equations method offers an alternative to gradient descent. This numerical method provides an exact solution, avoiding the iterative approach of gradient descent. It can be faster for certain problems but is more complex and will be covered in detail later.

## Extension of the Current Model

The linear regression model can be extended to include multiple features. For example, in a housing model, features could include size, age, number of bedrooms, and number of floors. However, a challenge arises when dealing with more than three dimensions, as visualization becomes difficult. To effectively manage multiple features, linear algebra concepts like matrices and vectors are employed, facilitating calculations and interpretations in higher-dimensional spaces.
