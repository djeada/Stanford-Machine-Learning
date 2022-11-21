## Overfitting with linear regression

* The same thing may happen with logistic regression.
* Sigmoidal function is an underfit.
* A high order polynomial, on the other hand, results in overfitting (high variance hypothesis).

![overfitting_logistic_regression](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/overfitting_logistic_regression.png)

## Cost function optimization for regularization

Penalize and make some of the $\theta$ parameters really small.


$$min \frac{1}{2m} \sum_{i=1}^{m}(h_{\theta}(x^{(i)} + y^{(i)})^2 + 1000 \theta_3^2 +  1000 \theta_4^2$$


So we simply end up with $\theta_3$ and $\theta_4$ being near to zero (because of the huge constants) and we're basically left with a quadratic function.

![optimization_regularization](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/optimization_regularization.png)


### Regularization
Small parameter values correlate to a simpler hypothesis (you effectively get rid of some of the terms). Overfitting is less likely with a simpler hypothesis.

$$J(\theta) = \frac{1}{2m} [ \sum_{i=1}^{m}(h_{\theta}(x^{(i)} - y^{(i)})^2 + \lambda \sum_{j=1}^{m} \theta_j^2] $$


$\lambda$ is the regularization parameter that controls a trade off between our two goals:


* Want to fit the training set well.
* Want to keep parameters small.


Later in the course, we'll look at various automated methods for selecting $\lambda$.


## Regularized linear regression

    while not converged:
      for j in [0, ..., n]:
          \theta_j := \theta_j - \alpha [\frac{1}{m} \sum_{i=1}^{m}(h_{\theta}(x^{(i)} + y^{(i)})x_j^{(i)} + \frac{\lambda}{m} \theta_j]
          
## Regularization with the normal equation

![regularized_normal_equation](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/regularized_normal_equation.png)

## Regularized logistic regression

Logistic regression cost function is as follows:

$$J(\theta) = \frac{1}{m} \sum_{i=1}^{m}[-y^{(i)}log(h_{\theta}(x^{(i)})) - (1-y^{(i)})log(1 - h_{\theta}(x^{(i)}))] + \frac{\lambda}{2m}\sum_{j=1}^{n}\theta_j^2$$

The gradient of the cost function is a vector where the j th element is defined as follows:

$$\frac{\partial}{\partial \theta_0} J(\theta) = \frac{1}{m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})x_j^{(i)} \quad for\;j=0$$

$$\frac{\partial}{\partial \theta_j} J(\theta) = (\frac{1}{m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})x_j^{(i)}) + \frac{\lambda}{m}\theta_j \quad for\;j\ge1$$

    while not converged:
      for j in [0, ..., n]:
          \theta_j := \theta_j - \alpha [\frac{1}{m} \sum_{i=1}^{m}(h_{\theta}(x^{(i)} + y^{(i)})x_j^{(i)} + \frac{\lambda}{m} \theta_j]
          
It appears to be the same as linear regression, except that the hypothesis is different.
