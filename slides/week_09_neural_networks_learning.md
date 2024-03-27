## Neural Networks: Learning and Classification

Neural networks, a core algorithm in machine learning, draw inspiration from the human brain's structure and function. They consist of layers containing interconnected nodes (neurons), each designed to perform specific computational tasks. Neural networks can tackle various classification problems, such as binary and multi-class classifications. The efficacy of these networks is enhanced through the adjustment of their parameters, guided by a cost function that quantifies the deviation between predictions and actual values.

### Classification Problems in Neural Networks

- **Training Set Representation:** Typically represented as $\{(x^1, y^1), (x^2, y^2), ..., (x^n, y^n)\}$, where $x^i$ is the $i^{th}$ input and $y^i$ is the corresponding target output.
- $L$: Total number of layers in the network.
- $s_l$: Number of units (excluding the bias unit) in layer $l$.

#### Example Network Architecture

![Example of Multi-Layer Neural Network](https://user-images.githubusercontent.com/37275728/201518449-ec13fac4-0716-4131-8e5e-f0405ce075a5.png)

In the illustrated network:

- There are 4 layers ($L=4$).
- The first layer ($s_1$) has 3 units.
- The second and third layers ($s_2$, $s_3$) each have 5 units.
- The fourth layer ($s_4$) has 4 units.

### Binary Classification

In binary classification, the neural network predicts a single output which can take one of two possible values (often represented as 0 or 1). Characteristics of binary classification in neural networks include:

- **Single Output Node:** The final layer produces a real number value, interpreted as a probability or classification decision.
- **Output Representation:** $k = 1$ (one output class) and $s_L = 1$ (one unit in the output layer).

### Multi-class Classification

Multi-class classification extends the network's capability to differentiate between multiple classes. Here, the target output $y$ is a vector representing the probability of each class. Characteristics include:

- **Multiple Output Classes:** $k$ distinct classifications are represented.
- **Output Vector:** The output $y$ is a $k$-dimensional vector of real numbers, where each element corresponds to a class.
- **Layer Configuration:** The output layer has $s_L = k$ units, each corresponding to one of the $k$ classes.

### Learning Process

The learning process in neural networks involves:

1. **Cost Function:** A generalization of the logistic regression cost function, which sums the error over all output units and classes. This function quantifies the difference between predicted and actual values.
2. **Gradient Computation:** Employing the chain rule to calculate the partial derivatives of the cost function with respect to each network parameter.
3. **Backpropagation Algorithm:** Used for efficiently computing these partial derivatives. The algorithm propagates the error signal backwards through the network, adjusting the parameters (weights and biases) to minimize the cost function.

### Neural Network Cost Function

The cost function for neural networks is an extension of the logistic regression cost function, adapted for multiple outputs. This function evaluates the performance of a neural network by measuring the difference between the predicted outputs and the actual values across all output units.

The cost function for a neural network with multiple outputs (where $K$ is the number of output units) is defined as:

$$J(\Theta) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K}[ y_k^{(i)} \log{(h_{\Theta}(x^{(i)}))}_k + (1- y_k^{(i)}) \log(1 - {(h_{\Theta} (x^{(i)}))}_k)] +  \frac{\lambda}{2m} \sum_{l=1}^{L-1} \sum_{i=1}^{s_l} \sum_{j=1}^{s_{l+1}} (\Theta^{(l)}_{ji})^2$$

- **$m$:** Number of training examples.
- **$K$:** Number of output units.
- **$L$:** Total number of layers in the network.
- **$s_l$:** Number of units in layer $l$.
- **$y_k^{(i)}$:** Actual value of the $k^{th}$ output unit for the $i^{th}$ training example.
- **$h_{\Theta}(x^{(i)})$:** Hypothesis function applied to the $i^{th}$ training example.
- **$\lambda$:** Regularization parameter to avoid overfitting.
- **$\Theta^{(l)}_{ji}$:** Weight from unit $i$ in layer $l$ to unit $j$ in layer $l+1$.

This function incorporates a sum over the $K$ output units for each training example, and also includes a regularization term to penalize large weights and prevent overfitting.

### Partial Derivative Terms

In order to optimize the cost function, we need to understand how changes in the weights $\Theta$ affect the cost. This is where the partial derivative terms come into play.

- **$\Theta^{(1)}_{ji}$:** Represents the weight from the $i^{th}$ unit in layer 1 to the $j^{th}$ unit in layer 2.
- **$\Theta^{(1)}_{10}$, $\Theta^{(1)}_{11}$, $\Theta^{(1)}_{21}$:** Specific weights that map from the input layer to the first and second units in the second layer, including the bias unit.

### Gradient Computation

Gradient computation in neural networks is achieved through a process known as backpropagation, which calculates the gradient of the cost function with respect to each weight in the network.

#### Forward Propagation Example

Consider a neural network with one training example $(x,y)$:

- **Layer 1 (Input Layer):** $a^{(1)} = x$ and $z^{(2)} = \Theta^{(1)}a^{(1)}$.
- **Layer 2 (Hidden Layer):** $a^{(2)} = g(z^{(2)})$, append $a^{(2)}_0$, and then $z^{(3)} = \Theta^{(2)}a^{(2)}$.
- **Layer 3 (Another Hidden Layer):** $a^{(3)} = g(z^{(3)})$, append $a^{(3)}_0$, and then $z^{(4)} = \Theta^{(3)}a^{(3)}$.
- **Output Layer:** $a^{(4)} = h_{\Theta}(x) = g(z^{(4)})$.

![Gradient Computation in Neural Networks](https://user-images.githubusercontent.com/37275728/201518441-7740e76d-9a6b-426f-98ad-85a5ff207a89.png)

Each layer's output ($a^{(l)}$) becomes the input for the next layer, with the activation function $g$ (typically a sigmoid or ReLU function) applied to the weighted sums. The final output $a^{(4)}$ is the hypothesis of the network for the given input $x$. The backpropagation algorithm then computes the gradient by propagating the error backwards from the output layer to the input layer, adjusting the weights $\Theta$ to minimize the cost function $J(\Theta)$.

### Backpropagation Algorithm

Backpropagation is a key algorithm in training neural networks, used for calculating the gradient of the cost function with respect to each parameter in the network. It involves propagating errors backward through the network, from the output layer to the input layer.

#### Calculating Error Terms (Deltas)

- **$\delta_j$ Vector:** Represents the error for each node $j$ in layer $l$. It is an $L \times 1$ vector, where $L$ is the total number of layers.
- **Last Layer Error ($\delta^{(L)}$):** For the last layer, the error is calculated as the difference between the network's output ($a^{(L)}$) and the actual value ($y$):

$$\delta^{(L)}_j = a^{(L)}_j - y_j$$

- **Error for Other Layers:** The error terms for the other layers are computed recursively using the errors from the subsequent layer:

$$\delta^{(l)}_j = (\Theta^{(l)}_j)^T \delta^{(l+1)} \cdot g'(z^{(l)}_j)$$

#### Accumulating Gradient ($\Delta$)

- **Partial Derivative Accumulation:** $\Delta^{(l)}_{ij}$ accumulates the partial derivatives of the cost function with respect to the weights $\Theta^{(l)}_{ij}$:

$$\Delta^{(l)}_{ij} := \Delta^{(l)}_{ij} + a^{(l)}_j \delta^{(l+1)}_i$$

- **Layer Notation:** $l$ denotes the layer, $j$ the node in layer $l$, and $i$ the error of the affected node in the subsequent layer $l+1$.

#### Calculating the Gradient

The gradient of the cost function $J(\Theta)$ with respect to the weights is given by:

$$
\frac{\partial}{\partial \Theta^{(l)}_{ij}}J(\Theta) = \begin{cases}
        \frac{1}{m} \Delta^{(l)}_{ij} + \lambda \Theta ^{(l)}_{ij} \quad &\text{if} \, j \neq 0 \\
        \frac{1}{m} \Delta^{(l)}_{ij} \quad &\text{if} \, j=0 \\
   \end{cases}
$$

### Gradient Checking

Due to the complexity of backpropagation, implementing it correctly is crucial. Gradient checking is a technique to validate the correctness of the computed gradients.

Steps in Gradient Checking:

1. **Function $J(\Theta)$:** Start with a function representing the cost.
2. **Compute Perturbed Weights:** Calculate $J(\Theta + \epsilon)$ and $J(\Theta - \epsilon)$, where $\epsilon$ is a small value.
3. **Slope Estimation:** The derivative is approximated by the slope of the straight line joining these two points.

![Gradient Checking Visualization](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/gradient_checking.png)

Gradient checking serves as a diagnostic tool to ensure that the backpropagation implementation is correct. However, it's computationally expensive and typically used only for debugging and not during regular training.
