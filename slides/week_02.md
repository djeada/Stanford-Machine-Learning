## Linear Regression
The previously described home price data example is an example of a supervised
learning regression problem.
Notation (used throughout the course)
* m = number of training examples.
* x′ s = input variables / features.
* y ′ s = output variable ”target” variables.
* (x, y) - single training example.
* (xi , y j ) - specific example (ith training example).

![house price table](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/house_price_table.png)

With our training set defined - how do we used it?
* Take training set.
* Pass into a learning algorithm.
* Algorithm outputs a function (h = hypothesis).
* This function takes an input (e.g. size of new house).
* Tries to output the estimated value of Y.
hθ (x) = θ0 + θ1 x
* Y is a linear function of x!
* θ0 is zero condition.
* θ1 is gradient.

Cost function
* We may use a cost function to determine how to fit the best straight line
to our data.
* We want to want to solve a minimization problem. Minimize (hθ (x) − y)2 .
* Sum this over the training set.

Example
For θ0 = 0 we have:

Data:
* θ1 = 1 and J(θ1 ) = 0.
* θ1 = 0.5 and J(θ1 ) = 0.58.
* θ1 = 0 and J(θ1 ) = 2.3.

![cost_function](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/cost_function.png)


The optimization objective for the learning algorithm is find the value of θ1
which minimizes J(θ1 ). So, here θ1 = 1 is the best value for θ1 .

A deeper insight into the cost function - simplified cost
function
The real cost function takes two variables as parameters! J(θ0 , θ1 ).
We can now generates a 3D surface plot where axis are:
* X = θ1 .
* Z = θ0 .
* Y = J(θ0 , θ1 ).

![surface_cost_function](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/surface_cost_function.png)

The best hypothesis is at the bottom of the bowl.
Instead of a surface plot we can use a contour figures/plots.
* Set of ellipses in different colors.
* Each colour is the same value of J(θ0 , θ1 ), but obviously plot to different
locations because θ1 and θ0 will vary.
* Imagine a bowl shape function coming out of the screen so the middle is
the concentric circles.

![contour_cost_function](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/contour_cost_function.png)


The best hypothesis is located in the center of the contour plot.
Gradient descent algorithm
* Begin with initial guesses, could be (0,0) or any other value.
* Continually change values of θ0 and theta1 to try to reduce J(θ0 , θ1 ).
* Continue until you reach a local minimum.
* Reached minimum is determined by the starting point.

![gradient_descent](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/gradient_descent.png)

![gradient_descent](https://user-images.githubusercontent.com/37275728/201476896-555ad8c4-8422-428b-937f-12cdf70d75bd.png)

Two key terms in the algorithm:
* α term
* Derivative term
Partial derivative vs. derivative
* Use partial derivative when we have multiple variables but only derive
with respect to one.
* Use derivative when we are deriving with respect to all the variables.
Derivative says:
* Let’s look at the slope of the line by taking the tangent at the point.
* As a result, going towards the minimum (down) will result in a nega-
tive derivative; nevertheless, alpha is always positive, thus J(θ1 ) will be
updated to a lower value.
* Similarly, as we progress up a slope, we increase the value of J(θ1 )

α term
* If it’s too small, it takes too long to converge.
* If it is too large, it may exceed the minimum and fail to converge.

When you get to a local minimum
* Gradient of tangent/derivative is 0
* So derivative term = 0
* α·0=0
* So θ1 = θ1 − 0.
* So θ1 remains the same.

Linear regression with gradient descent
Apply gradient descent to minimize the squared error cost function J(θ0 , θ1 ).

The linear regression cost function is always a convex function - always has a
single minimum. So gradient descent will always converge to global optima.

Two extension to the algorithm
Normal equation for numeric solution
* To solve the minimization problem we can solve it [minJ(θ0 , θ1 )] exactly
using a numeric method which avoids the iterative approach used by gra-
dient descent
Normal equations method.
* Can be much faster for some problems, but it is much more complicated
(will be covered in detail later).
We can learn with a larger number of features
* e.g. with houses: Size, Age, Number bedrooms, Number floors...
* Can’t really plot in more than 3 dimensions.
* Best way to get around with this is the notation of linear algebra (matrices
and vectors)
