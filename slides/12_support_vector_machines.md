# Support Vector Machines: Mathematical Insights

Support vector machines fit a decision score using a margin-based loss and a coefficient norm penalty. Comparing hinge with logistic loss explains their objectives. The SVM construction also leads to kernels that can represent nonlinear boundaries.

## Understanding Logistic Regression Cost Function

Logistic regression employs a hypothesis function defined as:

$$h_{\theta}(x) = \frac{1}{1 + e^{-\theta^T x}}$$

This function outputs the probability of $y = 1$ given $x$ and parameter $\theta$.

![sigmoid](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/sigmoid2.png)

For a given example where $y = 1$, ideally, $h_{\theta}(x)$ should be close to 1. The cost function for logistic regression is:

$$\text{Cost}(h_{\theta}(x), y) = -y \log(h_{\theta}(x)) - (1 - y) \log(1 - h_{\theta}(x))$$

![log_function](https://user-images.githubusercontent.com/37275728/201519577-c93854b4-1270-4082-9d9b-da0d543b0375.png)

When integrated over all samples, this becomes:

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m}[y^{(i)}\log(h_{\theta}(x^{(i)})) + (1 - y^{(i)})\log(1 - h_{\theta}(x^{(i)}))]$$

For signed margin $z=yf(x)$ with labels in $\{-1,1\}$, logistic loss is $\log(1+e^{-z})$. It grows approximately linearly for a very negative margin and decays toward zero for a very positive margin.

## Transitioning to SVM Cost Functions

SVMs modify this logistic cost function to a piecewise linear form. The idea is to use two functions, $cost_1(\theta^T x)$ for $y=1$ and $cost_0(\theta^T x)$ for $y=0$. These functions are linear approximations of the logistic function's cost. The SVM cost function becomes:

$$J(\theta) = C \sum_{i=1}^{m}[y^{(i)} cost_1(\theta^T x^{(i)}) + (1 - y^{(i)}) cost_0(\theta^T x^{(i)})] + \frac{1}{2} \sum_{j=1}^{n} \theta_j^2$$

The parameter $C$ plays a crucial role in SVMs. It balances the trade-off between the smoothness of the decision boundary and classifying the training points correctly.

- A large $C$ applies a higher penalty to misclassified examples, leading to a decision boundary that might have lower bias but higher variance, potentially causing overfitting.
- A small $C$ makes the decision boundary smoother, possibly with higher bias but lower variance, risking underfitting.

![svm_cost](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/svm_cost.png)

## Large Margin Intuition in SVMs

Support Vector Machines (SVMs) are designed to find a decision boundary not just to separate the classes, but to do so with the largest possible margin. This large margin intuition is key to understanding SVMs.

![large_dist](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/large_dist.png)

### The Concept of Margin in SVMs

When we talk about SVMs, we often refer to minimizing a function of the form `CA + B`. Let's break this down:

- **Minimizing A**: This part relates to minimizing the misclassification errors. If we set the parameter `C` to be very large, we prioritize minimizing these errors. In other words, if `C` is large, we aim to make `A` equal to zero. For this to happen:
  - If `y = 1`, we need to find a $\theta$ such that $\theta^Tx\geq1$.
  - If `y = 0`, we need to find a $\theta$ such that $\theta^Tx\leq-1$.
- **Minimizing B**: This part relates to maximizing the margin. The margin is the distance between the decision boundary and the nearest data point from either class. The optimization problem can be expressed as:

$$
\begin{aligned}
\text{minimize}\ \frac{1}{2} \sum_{j=1}^{n} \theta_j^2 \\
\text{subject to}\ \theta^Tx^{(i)} \geq 1\ \text{if}\ y^{(i)}=1 \\
\theta^Tx^{(i)} \leq -1\ \text{if}\ y^{(i)}=0
\end{aligned}
$$

### Visualization of Large Margin Classification

In the visualization, the green and magenta lines are potential decision boundaries that might be chosen by a method like logistic regression. However, these lines might not generalize well to new data. The black line, chosen by the SVM, represents a stronger separator. It is chosen because it maximizes the margin, the minimum distance to any of the training samples.

### Understanding the Decision Boundary in SVMs

Assuming we have only two features and $\theta_0 = 0$, we can rewrite the minimization of `B` as:

$$\frac{1}{2}(\theta_1^2 + \theta_2^2) = \frac{1}{2}(\sqrt{\theta_1^2 + \theta_2^2})^2 = \frac{1}{2}||\theta||^2$$

This is essentially minimizing the norm of the vector $\theta$. Now, consider what $\theta^Tx$ represents in this context. If we have a positive training example and we plot $\theta$ on the same axis, we are interested in the inner product of these two vectors, denoted as `p`.

- `p` is actually `p^i`, representing the projection of the training example `i` on the vector $\theta$.
- The conditions for the classification become:

$$
\begin{aligned}
p^{(i)} \cdot ||\theta|| &\geq 1\ \text{if}\ y^{(i)}=1 \\
p^{(i)} \cdot ||\theta|| &\leq -1\ \text{if}\ y^{(i)}=0
\end{aligned}
$$

![svm_vectors](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/svm_vectors.png)

## Adapting SVM to Non-linear Classifiers

Support Vector Machines (SVMs) can be adapted to find non-linear boundaries, which is crucial for handling datasets where the classes are not linearly separable.

### Non-linear Classification

Consider a training set where a linear decision boundary is not sufficient. The goal is to find a non-linear boundary that can effectively separate the classes.

![Example of a Non-linear Boundary](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/non_linear_boundary.png)

### Introducing Landmarks and Features

1. **Defining Landmarks**: We start by defining landmarks in our feature space. Landmarks ($l^1$, $l^2$, and $l^3$) are specific points in the feature space, chosen either manually or by some heuristic.

    ![Landmarks in Feature Space](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/landmarks.png)

2. **Kernel Functions**: A kernel is a function that computes the similarity between each feature $x$ and the landmarks. For example, the Gaussian kernel for a landmark $l^1$ is defined as:

   $$f_1 = k(x, l^1) = \exp\left(- \frac{||x - l^{(1)}||^2}{2\sigma^2}\right)$$

    - A large $\sigma^2$ leads to smoother feature variation (higher bias, lower variance).
    - A small $\sigma^2$ results in abrupt feature changes (low bias, high variance).

3. **Prediction Example**: Consider predicting the class of a new point (e.g., the magenta dot in the figure). Using our kernel functions and a specific set of $\theta$ values, we can compute the classification.

    ![Evaluating a New Point](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/landmarks_magneta.png)

    For a point close to $l^1$, $f_1$ will be close to 1, and others will be closer to 0. Hence, for $\theta_0 = -0.5, \theta_1 = 1, \theta_2 = 1, \theta_3 = 0$, the prediction will be 1.

### Choosing Landmarks

- Landmarks are often chosen to be the same as the training examples, resulting in `m` landmarks for `m` training examples.
- Each example's feature set is evaluated based on its proximity to each landmark using the chosen kernel function.

### Different Kernels

- **Linear Kernel**: Equivalent to no kernel, predicts $y = 1$ if $(\theta^T x) \geq 0$.
- **Gaussian Kernel**: Useful for creating complex non-linear boundaries.
- Other types of kernels include Polynomial, String, Chi-squared, and Histogram Intersection kernels.

### Logistic Regression vs. SVM in Non-linear Classification

- **High Feature Count (n) Compared to Examples (m)**: Use logistic regression or SVM with a linear kernel.
- **Small n, Intermediate m**: Gaussian kernel with SVM is suitable.
- **Small n, Large m**: SVM with Gaussian kernel may be slow; logistic regression or SVM with a linear kernel is preferred.
- **SVM's Power**: The ability to use different kernels makes SVMs versatile for learning complex non-linear functions.
- **Convex Optimization**: The standard objective with a valid positive-semidefinite kernel is convex. Numerical accuracy depends on solver tolerance; equal optimal objective values do not require identical dual coefficients.

## Hinge Loss and Geometric Margin

Use labels $y_i\in\{-1,1\}$ and score $f(x)=w^\top x+b$. The soft-margin objective is

$$
\frac12\|w\|^2+C\sum_i\max(0,1-y_if(x_i)).
$$

A correctly classified point still incurs hinge loss when its signed score lies between zero and one. Loss reaches zero at the unit functional margin. The intercept is unpenalized in this formulation.

The signed distance to the boundary is $y_i(w^\top x_i+b)/\|w\|$. Under hard-margin constraints $y_if(x_i)\geq1$, maximizing minimum distance is equivalent to minimizing $\|w\|^2$. The margin-plane separation is $2/\|w\|$. Rescaling w and b leaves the boundary unchanged but changes functional margins, so the unit constraints fix the scale.

For negative points $-2,-1$ and positive points $1,2$, the solution $w=1,b=0$ has closest points $-1$ and one. Each lies one unit from the boundary; the margin planes are two units apart.

## Verify a Linear Fit and Reconstruct a Kernel Decision

```python
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC
X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
y = np.array([-1, -1, 1, 1])
linear = SVC(kernel='linear', C=1000000.0, tol=1e-10).fit(X, y)
np.testing.assert_allclose(linear.coef_, [[1.0]], atol=1e-08)
np.testing.assert_allclose(linear.intercept_, [0.0], atol=1e-08)
np.testing.assert_allclose(np.sort(linear.support_vectors_[:, 0]), [-1.0, 1.0])
geometric_margin = 1 / np.linalg.norm(linear.coef_)
np.testing.assert_allclose(geometric_margin, 1.0)
points = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
labels = np.array([-1, 1, 1, -1])
gamma = 2.0
rbf = SVC(kernel='rbf', gamma=gamma, C=10.0, tol=1e-10).fit(points, labels)
kernel = np.exp(-gamma * np.sum((points[:, None, :] - rbf.support_vectors_[None, :, :]) ** 2, axis=2))
manual = kernel @ rbf.dual_coef_[0] + rbf.intercept_[0]
np.testing.assert_allclose(kernel, rbf_kernel(points, rbf.support_vectors_, gamma=gamma))
np.testing.assert_allclose(manual, rbf.decision_function(points), atol=1e-10)
np.testing.assert_array_equal(rbf.predict(points), labels)
assert np.linalg.eigvalsh(rbf_kernel(points, gamma=gamma)).min() > 0.0
print('linear w/b/margin:', linear.coef_[0], linear.intercept_, geometric_margin)
print('RBF XOR decisions:', np.round(manual, 6))
```

The linear check compares the coefficient, intercept, support vectors, and margin with the hand solution. The RBF check solves a nonlinear XOR fixture, then independently reconstructs its scores from support vectors and stored dual coefficients.

The RBF formula $k(x,z)=e^{-\gamma\|x-z\|^2}$ matches the course parameterization at $\gamma=1/(2\sigma^2)$. With gamma two, points one unit apart have similarity $e^{-2}\approx0.135335$. Raising gamma shortens its distance scale. Scaling coordinates changes this meaning and belongs inside validation folds.

## Similarity Features and a Kernel SVM Use Different Norms

A kernel decision is $f(x)=\sum_i\alpha_i y_i k(x_i,x)+b$. Support vectors have nonzero dual coefficients. The kernel supplies inner products in a feature space, requiring positive-semidefinite Gram matrices for the standard construction.

Creating similarities to landmarks and fitting an ordinary linear model to that vector defines another model. A kernel SVM penalizes the feature-space norm, which becomes a quadratic form in expansion coefficients. Penalizing only their ordinary Euclidean norm generally changes the objective. Matching a similarity formula alone does not establish equivalent fitted models.

Large C makes violations more expensive relative to the norm penalty. It can amplify sensitivity to mislabeled observations and does not make nonseparable data separable. Validation should compare complete scaled procedures. An SVM score measures signed decision strength; interpreting it as probability requires an additional fitted probability model.

## Reference

These notes are based on the free video lectures offered by Stanford University, led by Professor Andrew Ng. These lectures are part of the renowned Machine Learning course available on Coursera. For more information and to access the full course, visit the [Coursera course page](https://www.coursera.org/learn/machine-learning).
