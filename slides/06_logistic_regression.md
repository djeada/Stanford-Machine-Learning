## Logistic Regression

Logistic regression is a statistical method used for classification in machine learning. Unlike linear regression, which predicts continuous values, logistic regression predicts discrete outcomes, like classifying an email as spam or not spam.

### Classification 
Yields discrete values (e.g., 0 or 1, representing classes).

Examples:

- Email: Spam (1) or Not Spam (0).
- Online Transaction: Fraudulent (1) or Not Fraudulent (0).
- Tumor Diagnosis: Malignant (1) or Benign (0).

### Logistic Regression vs Linear Regression
Applying linear regression to classification tasks, like cancer diagnosis, may not yield effective results, especially when the data doesn't fit well into a linear model.

![cancer_classification](https://user-images.githubusercontent.com/37275728/201496614-36ec47d4-437e-4d25-82bf-27289489a5a7.png)

### Hypothesis Representation

- Classifier output should be between 0 and 1 (probability).
- Hypothesis $h_{\theta}(x) = g(\theta^Tx)$.
- $g(z)$ is the sigmoid or logistic function:

$$
g(z) = \frac{1}{1 + e^{-z}}
$$

Hypothesis Equation:

$$
h_{\theta}(x) = \frac{1}{1+e^{-\theta^Tx}}
$$

- The output of $h_{\theta}(x)$ is interpreted as the probability of the positive class given the input $x$.
  $$ h_{\theta}(x) = P(y=1|x\ ;\ \theta) $$
- **Example**: If $h_{\theta}(x) = 0.7$ for a tumor, it implies a 70% chance of the tumor being malignant.

### Sigmoid Function
Visualizes how $h_{\theta}(x)$ translates a linear combination of inputs into a probability:
  
![sigmoid](https://user-images.githubusercontent.com/37275728/201496643-38a45685-61a5-4af4-bf24-2acaa22ef1ff.png)

### Decision Boundary in Logistic Regression

The decision boundary in logistic regression is critical for classification tasks. It separates the different classes based on the probability calculated using the sigmoid function.

#### Linear Decision Boundary

- **Principle**: Predict $y = 1$ if the probability is greater than 0.5, else predict $y = 0$.
- **Hypothesis**: $h_{\theta}(x) = g(\theta^T x)$, where $g$ is the sigmoid function.
- **Predicting $y = 1$**: Occurs when $\theta^T x \geq 0$.
- **Predicting $y = 0$**: Occurs when $\theta^T x \leq 0$.

#### Example of a Linear Decision Boundary

Hypothesis: 

$$
h_{\theta}(x) = g(\theta_0 + \theta_1x_1 + \theta_2x_2)
$$

Theta Vector:

$$
\theta = \begin{bmatrix}
-3 \\
1  \\
1
\end{bmatrix}
$$

Condition for $y = 1$:

$$
-3 + x_1 + x_2 \geq 0
$$

Hence, the decision boundary is a straight line: $x_2 = -x_1 + 3$.

![linear_decision_boundary](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/linear_decision_boundary.png)

#### Non-linear Decision Boundaries
- **Purpose**: To fit more complex, non-linear datasets.
- **Approach**: Introduce polynomial terms in the hypothesis.

#### Example of a Non-linear Decision Boundary
Hypothesis: 

$$
h_{\theta}(x) = g(\theta_0 + \theta_1x_1 + \theta_2x_1^2 + \theta_3x_2^2)
$$

Theta Vector:

$$
\theta = \begin{bmatrix}
-1 \\
0  \\
1  \\
1
\end{bmatrix}
$$

Condition for $y = 1$:

$$
x_1^2 + x_2^2 \geq 1
$$

This forms a circular decision boundary with radius 1 around the origin: $x_1^2 + x_2^2 = 1$.

![non_linear_decision_boundary](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/non_linear_decision_boundary.png)

### Cost Function for Logistic Regression

Logistic regression uses a different cost function compared to linear regression, tailored to the classification setting.

#### Training Set Representation
Consider a training set of $m$ examples:

$$ \{(x^{(1)}, y^{(1)}), ..., (x^{(m)}, y^{(m)})\} $$

where

$$  
x = \begin{bmatrix}
    x_0 \\
    x_1 \\
    ... \\
    x_n
\end{bmatrix}
$$

with $x_0 = 1$ and $y$ being either 0 or 1.

#### Linear Regression Cost Function
In linear regression, the cost function $J(\theta)$ is defined as:

$$ J(\theta) = \frac{1}{2m} \sum_{i=1}^{m}(h_{\theta}(x^{(i)}) - y^{(i)})^2 $$

#### Defining Cost for Logistic Regression
For logistic regression, we define a different "cost" function:

$$ cost(h_{\theta}(x^{(i)}), y^{(i)}) = \frac{1}{2} (h_{\theta}(x^{(i)}) - y^{(i)})^2 $$

Redefining $J(\theta)$:

$$ J(\theta) = \frac{1}{m} \sum_{i=1}^{m}cost(h_{\theta}(x^{(i)}), y^{(i)}) $$

This cost function for logistic regression is not convex, leading to potential issues with local optima.

#### Logistic Regression Cost Function
The logistic regression cost function is defined as:

$$
cost(h_{\theta}(x), y) = \begin{cases}
    -\log(h_{\theta}(x))     & \text{if } y=1 \\
    -\log(1 - h_{\theta}(x)) & \text{if } y=0
\end{cases}
$$

Then, the overall cost function $J(\theta)$ becomes:

$$J(\theta) = \frac{1}{m} \sum_{i=1}^{m}[-y^{(i)}\log(h_{\theta}(x^{(i)})) - (1-y^{(i)})\log(1 - h_{\theta}(x^{(i)}))]
$$

#### Gradient of the Cost Function
The gradient of $J(\theta)$ for logistic regression is:

$$
\frac{\partial}{\partial \theta_j} J(\theta) = \frac{1}{m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})x_j^{(i)}
$$

Note: While this gradient looks identical to that of linear regression, the formulae differ due to the different definitions of $h_{\theta}(x)$ in linear and logistic regression.

### Multiclass Classification Problems

Logistic regression can be extended to handle multiclass classification problems through the "one-vs-all" (or "one-vs-rest") method.

#### One-vs-All Approach
The one-vs-all strategy involves training multiple binary classifiers, each focused on distinguishing one class from all other classes.

#### Visualization of Multiclass Classification
Consider a dataset with three classes: triangles, crosses, and squares.

![multiclass_classification](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/multiclass_classification.png)

#### Implementing One-vs-All
The process involves splitting the training set into separate binary classification problems:

1. **Triangle vs Others**: Train a classifier $h_{\theta}^{(1)}(x)$ to distinguish triangles (1) from crosses and squares (0).
2. **Crosses vs Others**: Train another classifier $h_{\theta}^{(2)}(x)$ to distinguish crosses (1) from triangles and squares (0).
3. **Squares vs Others**: Lastly, train a classifier $h_{\theta}^{(3)}(x)$ to distinguish squares (1) from crosses and triangles (0).

![one_vs_all](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/one_vs_all.png)

#### Classification Decision

- When classifying a new example, compute the probability that it belongs to each class using the respective classifiers.
- The class with the highest probability is chosen as the prediction.

## Reference

These notes are based on the free video lectures offered by Stanford University, led by Professor Andrew Ng. These lectures are part of the renowned Machine Learning course available on Coursera. For more information and to access the full course, visit the [Coursera course page](https://www.coursera.org/learn/machine-learning).
