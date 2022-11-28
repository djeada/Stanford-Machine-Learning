## Linear Regression
The previously described home price data example is an example of a supervised learning regression problem.

### Notation (used throughout the course)
* m = number of training examples.
* $x$ = input variables / features.
* $y$ = output variable ”target” variables.
* $(x, y)$ - single training example.
* $(x^i, y^i)$ - specific example (ith training example).

![house price table](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/house_price_table.png)

### With our training set defined - how do we use it?
* Take training set.
* Pass into a learning algorithm.
* Algorithm outputs a function (h = hypothesis).
* This function takes an input (e.g. size of new house).
* Tries to output the estimated value of Y.

$$h_{\theta}(x) = \theta_0 + \theta_1x$$

* Y is a linear function of x!
* $\theta_0$ is zero condition.
* $\theta_1$ is gradient.

### Cost function
* We may use a cost function to determine how to fit the best straight line
to our data.
* We want to want to solve a minimization problem. Minimize $(h_{\theta}(x) - y)^2$.
* Sum this over the training set.

$$J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{m}^{i=1}(h_{\theta}(x^{(i)}) - y^{(i)})^2$$

For $\theta_0 = 0$ we have:

$$h_{\theta}(x) = \theta_1x\quad and \quad J(\theta_1) = \frac{1}{2m} \sum_{m}^{i=1}(h_{\theta}(x^{(i)}) - y^{(i)})^2$$

Data:
* $\theta_1 = 1$ and $J(\theta_1)= 0$.
* $\theta_1 = 0.5$ and $J(\theta_1)= 0.58$.
* $\theta_1 = 0$ and $J(\theta_1)= 2.3$.

![cost_function](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/cost_function.png)

The optimization objective for the learning algorithm is find the value of θ1
which minimizes J(θ1 ). So, here θ1 = 1 is the best value for θ1 .

### A deeper insight into the cost function - simplified cost function

The real cost function takes two variables as parameters! $J(\theta_0, \theta_1)$.
We can now generates a 3D surface plot where axis are:

* $X = \theta_1$.
* $Z = \theta_0$.
* $Y = J(\theta_0,\theta_1)$.
  
![surface_cost_function](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/surface_cost_function.png)

The best hypothesis is at the bottom of the bowl.
Instead of a surface plot we can use a contour figures/plots.
* Set of ellipses in different colors.
* Each colour is the same value of $J(\theta_0,\theta_1)$, but obviously plot to different
locations because θ1 and θ0 will vary.
* Imagine a bowl shape function coming out of the screen so the middle is
the concentric circles.

![contour_cost_function](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/contour_cost_function.png)


The best hypothesis is located in the center of the contour plot.

## Gradient descent algorithm

    \theta = [0, 0]
    while not converged:
      for j in [0, 1]:
          \theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta_0, \theta_1)
      

* Begin with initial guesses, could be (0,0) or any other value.
* Repeatedly change values of $\theta_0$ and $\theta_1$ to try to reduce  $J(\theta_0, \theta_1)$.
* Continue until you reach a local minimum.
* Reached minimum is determined by the starting point.

![gradient_descent](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/gradient_descent.png)

![gradient_descent](https://user-images.githubusercontent.com/37275728/201476896-555ad8c4-8422-428b-937f-12cdf70d75bd.png)

Two key terms in the algorithm:
* $\alpha$ term.
* Derivative term.

### Partial derivative vs. derivative
* Use partial derivative when we have multiple variables but only derive with respect to one.
* Use derivative when we are deriving with respect to all the variables.

Derivative says:
* Let’s look at the slope of the line by taking the tangent at the point.
* As a result, going towards the minimum (down) will result in a negative derivative; nevertheless, alpha is always positive, thus $J(\theta_1)$ will be updated to a lower value.
* Similarly, as we progress up a slope, we increase the value of $J(\theta_1)$.

$\alpha$ term
* If it’s too small, it takes too long to converge.
* If it is too large, it may exceed the minimum and fail to converge.

When you get to a local minimum
* Gradient of tangent/derivative is 0.
* So derivative term = 0.
* $\alpha \cdot 0 = 0$.
* So $\theta_1 = \theta_1 - 0$.
* So $\theta_1$ remains the same.
  
## Linear regression with gradient descent
Apply gradient descent to minimize the squared error cost function $J(\theta_0, \theta_1)$.

$$\frac{\partial}{\partial \theta_j} J(\theta_0, \theta_1) = \frac{\partial}{\partial \theta_j} \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})^2$$

$$= \frac{\partial}{\partial \theta_j} \frac{1}{2m} \sum_{i=1}^{m} (\theta_0 + \theta_1x^{(i)} - y^{(i)})^2$$

For each case, we must determine the partial derivative:

$$j=0:\frac{\partial}{\partial \theta_0} J(\theta_0, \theta_1)=\frac{\partial}{\partial \theta_0} \frac{1}{m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})$$

$$j=1:\frac{\partial}{\partial \theta_1} J(\theta_0, \theta_1)=\frac{\partial}{\partial \theta_1} \frac{1}{m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})x^{(i)}$$

## Normal equations method
* To solve the minimization problem we can solve it $[ min J(\theta_0, \theta_1) ]$ exactly using a numerical method which avoids the iterative approach used by gradient descent.
* Can be much faster for some problems, but it is much more complicated (will be covered in detail later).

We can learn with a larger number of features
* e.g. with houses: Size, Age, Number bedrooms, Number floors...
* Can’t really plot in more than 3 dimensions.
* Best way to get around with this is the notation of linear algebra (matrices and vectors)
