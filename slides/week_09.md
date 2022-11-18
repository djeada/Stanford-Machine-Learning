## Types of classification problems with NNs

* Training set is ${(x^1, y^1), (x^2, y^2), ..., (x^n, y^n)}$.
* $L$ = number of layers in the network.
* $s_l$ = number of units (not counting bias unit) in layer $l$.

![big_multi_layer](https://user-images.githubusercontent.com/37275728/201518449-ec13fac4-0716-4131-8e5e-f0405ce075a5.png)

So here we have:

* $L=4$.
* $s_1 = 3$.
* $s_2 = 5$.
* $s_3 = 5$.
* $s_4 = 4$.


### Binary classification

* 1 output (0 or 1).
* So single output node - value is going to be a real number.
* $k = 1$.
* $s_l = 1$


### Multi-class classification

* $k$ distinct classifications.
* $y$ is a $k$-dimensional vector of real numbers.
* $s_l = k$

## Cost function

Logistic regression cost function is as follows:

$$J(\theta) = -\frac{1}{m} [\sum_{i=1}^{m} y^{(i)} log h_{\theta}(x^{(i)}) + (1- y^{(i)})log(1 - h_{\theta}(x^{(i)}))] +  \frac{\lambda}{2m} \sum_{j=1}^{m} \theta_j^2$$


For neural networks our cost function is a generalization of this equation above, so instead of one output we generate $k$ outputs:

$$J(\Theta) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K}[ y_k^{(i)} \log{(h_{\theta}(x^{(i)}))}_k + (1- y_k^{(i)}) \log(1 - {(h_{\theta} (x^{(i)}))}_k)] +  \frac{\lambda}{2m} \sum_{l=1}^{L-1} \sum_{i=1}^{s_l} \sum_{j=1}^{s_l+1} (\Theta^{(l)}_{ji})^2$$

* Our cost function now outputs a $k$ dimensional vector.
* Costfunction $J(\Theta)$ is $-1/m$ times a sum of a similar term to which we had for logic regression.
* But now this is also a sum from $k = 1$ through to $K$ ($K$ is number of output nodes).
* Summation is a sum over the $k$ output units - i.e. for each of the possible classes.
* We don't sum over the bias terms (hence starting at 1 for the summation).

### Partial derivative terms


* $\Theta^{(1)}$ is the matrix of weights which define the function mapping from layer 1 to layer 2.
* $\Theta^{(1)}_{10}$  is the real number parameter which you multiply the bias unit (i.e. 1) with for the bias unit input into the first unit in the second layer.
* $\Theta^{(1)}_{11}$ is the real number parameter which you multiply the first (real) unit with for the first input into the first unit in the second layer.
* $\Theta^{(1)}_{21}$  is the real number parameter which you multiply the first (real) unit with for the first input into the second unit in the second layer.


### Gradient computation


* One training example.
* Imagine we just have a single pair (x,y) - entire training set.
* The following is how the forward propagation method works:
* Layer 1: $a^{(1)} = x$ and $z^{(2)} = \Theta^{(1)}a^{(1)}$.
* Layer 2: $a^{(2)} = g(z^{(2)}) + a^{(2)}_0$ and $z^{(3)} = \Theta^{(2)}a^{(2)}$.
* Layer 3: $a^{(3)} = g(z^{(3)}) + a^{(3)}_0$ and $z^{(4)} = \Theta^{(3)}a^{(3)}$.
* Ouptut: $a^{(4)} = h_{\Theta}(x) = g(z^{(4)})$.

![gradient_computing](https://user-images.githubusercontent.com/37275728/201518441-7740e76d-9a6b-426f-98ad-85a5ff207a89.png)

### Calculate backpropagation

* $\delta_j$ is $Lx1$ vector.
* First we compute the LAST element: $\delta^{(L)}_j = a^{(L)}_j - y_j$.
* Value of each element is based on the value of the next element:

  $$\delta^{(l)}_j = (\Theta^{(l)}_j)^T\delta^{(l+1)}_j \cdot g'(z^{(l)}_j)$$

* Finally, use $\Delta$ to accumulate the partial derivative terms:

$$\Delta^{(l)}_{ij} := \Delta^{(l)}_{ij} + a^{(l)}_j\delta^{(l+1)}_i$$

* $l$ = layer.
* $j$ = node in that layer.
* $i$ = the error of the affected node in the target layer

$$
\frac{\partial}{\partial \Theta^{(l)}_{ij}}J(\Theta) = \begin{cases}
          \frac{1}{m} \Delta^{(l)}_{ij} + \lambda \Theta ^{(l)}_{ij} \quad &\text{if} \, j \neq 0 \\
          \frac{1}{m} \Delta^{(l)}_{ij} \quad &\text{if} \, j=0 \\
     \end{cases}
$$

### Gradient checking

Backpropagation contains a lot of details, and tiny flaws can break it.
As a result, employing a numerical approach to verify the gradient can aid in the quick diagnosis of a problem.

* Have a function $J(\Theta)$.
* Compute $\Theta + \epsilon$.
* Compute $\Theta - \epsilon$.
* Join them by a straight line.
* Use the slope of that line as an approximation to the derivative.

![gradient_checking](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/gradient_checking.png)
