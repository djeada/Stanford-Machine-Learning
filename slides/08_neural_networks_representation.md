# Neural Networks Introduction

A neural network composes parameterized transformations of its input. A hidden layer can learn combinations of features, and a nonlinear activation prevents the entire network from collapsing into one linear map. The appropriate architecture depends on the task, sample size, and computational constraints; a linear classifier can remain effective with thousands of features.

A neural network comprises an intricate architecture of neurons, analogous to the brain's neural cells, linked through synaptic connections. These connections facilitate the transmission and processing of information across the network. The basic structure of a neural network includes:

- **Input Layer:** Serves as the entry point for data into the network.
- **Hidden Layers:** Intermediate layers that perform computations and feature transformations.
- **Output Layer:** Produces the final output of the network.

Each neuron in these layers applies a weighted sum to its inputs, followed by a nonlinear activation function. The weights of these connections are represented by the matrix $\theta$, while the constant bias input is denoted as $x_0=1$, with its learned bias coefficient included in the weights. These parameters are learned during the training process, enabling the network to capture complex patterns and relationships within the data.

## Mathematical Representation

Consider a neural network with $L$ layers, each layer $l$ having $s_l$ units (excluding the bias unit). The activation of unit $i$ in layer $l$ is denoted as $a_i^{(l)}$. The activation function applied at each neuron is usually a nonlinear function like the sigmoid or ReLU function. The cost function for a neural network, often a function like cross-entropy or mean squared error, is minimized during the training process to adjust the weights $\theta$.

## Significance in Computer Vision

Neural networks particularly shine in domains like computer vision, where data often involves high-dimensional input spaces. For instance, an image with a resolution of 50x50 pixels, considering only grayscale values, translates to 2,500 features. If we incorporate color channels (RGB), the feature space expands to 7,500 dimensions. Linear methods can process such vectors efficiently, but a raw-pixel linear boundary has limited representational structure. Neural networks introduce learned nonlinear features, while convolutional layers can share local filters across image positions. These are modeling choices, not a guarantee that a network will outperform a linear baseline.

## Neuroscience Inspiration

The biological analogy motivates names such as neuron, connection, and activation. The implemented learning rule is specified by matrix operations, a loss, and an optimizer. Those definitions let us calculate predictions and derivatives directly, without assuming a general model of biological cognition.

## Model Representation I

### Neuron Model in Biology

In biological terms, a neuron consists of three main parts:

- **Cell Body:** Central part of a neuron where inputs are aggregated.
- **Dendrites:** Input wires which receive signals from other neurons.
- **Axon:** Output wire that transmits signals to other neurons.

Biological neurons process signals through a combination of electrical and chemical means, sending impulses (or spikes) along the axon as a response to input stimuli received through dendrites.

![Biological Neuron Diagram](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/neuron.png)

### Artificial Neural Network: Neuron Representation

In artificial neural networks, the neuron or 'node' functions similarly to its biological counterpart but in a simplified and abstracted manner:

- **Inputs:** Received through 'input wires', akin to dendrites.
- **Computation:** Performed by a logistic unit within the neuron.
- **Output:** Sent down 'output wires', similar to the axon.

Each neuron in an artificial neural network processes its inputs using a weighted sum and an activation function. The result is then passed on to subsequent neurons or to the output layer.

![Artificial Neuron Diagram](https://user-images.githubusercontent.com/37275728/201517992-cdc14304-2af9-4821-bcae-71caa1a62663.png)

### Mathematical Model of a Neuron

Consider a neuron with inputs represented as a vector $x$, where $x_0$ is the bias unit:

$$
x = \begin{bmatrix}
x_{0} \\
x_{1} \\
x_2   \\
x_3
\end{bmatrix}
$$

And the corresponding weights of the neuron are denoted by $\theta$:

$$
\theta  = \begin{bmatrix}
\theta_{0} \\
\theta_{1} \\
\theta_2   \\
\theta_3
\end{bmatrix}
$$

In this representation, $x_0$ is the bias unit that helps in shifting the activation function, and $\theta$ represents the weights of the model.

### Layers in a Neural Network

- **Input Layer:** The initial layer that receives input data.
- **Hidden Layers:** Intermediate layers where data transformations occur. The activations within these layers are not directly observable.
- **Output Layer:** Produces the final output based on the computations performed by the network.

The connectivity pattern in a neural network typically involves each neuron in one layer being connected to all neurons in the subsequent layer.

![Hidden Layer Representation](https://user-images.githubusercontent.com/37275728/201517995-ff2af22c-ea22-4be9-9bfc-b7e6c771d69c.png)

### Activation and Output Computation

The activation $a^{(2)}_i$ of the $i^{th}$ neuron in the 2nd layer is calculated as a function $g$ (such as the sigmoid function) of a weighted sum of inputs:

$$
a^{(2)}_1 = g(\Theta^{(1)}_{10}x_0+\Theta^{(1)}_{11}x_1+\Theta^{(1)}_{12}x_2+\Theta^{(1)}_{13}x_3)
$$

$$
a^{(2)}_2 = g(\Theta^{(1)}_{20}x_0+\Theta^{(1)}_{21}x_1+\Theta^{(1)}_{22}x_2+\Theta^{(1)}_{23}x_3)
$$

$$
a^{(2)}_3 = g(\Theta^{(1)}_{30}x_0+\Theta^{(1)}_{31}x_1+\Theta^{(1)}_{32}x_2+\Theta^{(1)}_{33}x_3)
$$

The hypothesis function $h_{\Theta}(x)$ for a neural network is the output of the network, which in turn is the activation of the output layer's neurons:

$$
h_{\Theta}(x) = g(\Theta^{(2)}_{10}a^{(2)}_0+\Theta^{(2)}_{11}a^{(2)}_1+\Theta^{(2)}_{12}a^{(2)}_2+\Theta^{(2)}_{13}a^{(2)}_3)
$$

## Model Representation in Neural Networks II

Neural networks process large amounts of data, necessitating efficient computation methods. Vectorization is a key technique used to achieve this efficiency. It allows for the simultaneous computation of multiple operations, significantly speeding up the training and inference processes in neural networks.

### Defining Vectorized Terms

To illustrate vectorization, consider the computation of the activation for neurons in a layer. The activation of the $i^{th}$ neuron in layer 2, $a^{(2)}_i$, is based on a linear combination of inputs followed by a nonlinear activation function $g$ (e.g., sigmoid function). This can be represented as:

$$z^{(2)}_i = \Theta^{(1)}_{i0}x_0+\Theta^{(1)}_{i1}x_1+\Theta^{(1)}_{i2}x_2+\Theta^{(1)}_{i3}x_3$$

Hence, the activation $a^{(2)}_i$ is given by:

$$a^{(2)}_i = g(z^{(2)}_i)$$

### Vector Representation

Inputs and activations can be represented as vectors:

$$
x = \begin{bmatrix}
x_{0} \\
x_{1} \\
x_2   \\
x_3
\end{bmatrix}
$$

$$
z^{(2)} = \begin{bmatrix}
z^{(2)}_1 \\
z^{(2)}_2 \\
z^{(2)}_3
\end{bmatrix}
$$

- **$z^{(2)}$:** Vector representing the linear combinations for each neuron in layer 2.
- **$a^{(2)}$:** Vector representing the activations for each neuron in layer 2, calculated by applying $g()$ to each element of $z^{(2)}$.
- **Hidden Layers:** Middle layers in the network, which transform the inputs in a non-linear way.
- **Forward Propagation:** The process where input values are fed forward through the network, from input to output layer, producing the final result.

### Architectural Flexibility

Neural network architectures can vary in complexity:

- **Variation in Node Count:** The number of neurons in each layer can be adjusted based on the complexity of the task.
- **Number of Layers:** Additional layers can be added to create deeper networks, which are capable of learning more complex patterns.

![Multi-Layer Neural Network](https://user-images.githubusercontent.com/37275728/201517998-e5f9f245-a6f1-4aed-8a58-fcb0178f38c4.png)

In the above example, layer 2 has three hidden units, and layer 3 has two hidden units. By adjusting the number of layers and nodes, neural networks can model complex nonlinear hypotheses, enabling them to tackle a wide range of problems from simple linear classification to complex tasks in computer vision and natural language processing.

## Neural Networks for Logical Functions: AND and XNOR

### The AND Function

The AND function is a fundamental logical operation that outputs true only if both inputs are true. In neural networks, this can be modeled using a single neuron with appropriate weights and a bias.

![AND Function Graphical Representation](https://user-images.githubusercontent.com/37275728/201518002-72b41fb7-ca3f-4612-aa65-c34f58138737.png)

Let's define the bias unit as $x_0 = 1$. We can represent the weights for the AND function in the vector $\Theta^{(1)}_1$:

$$
\Theta^{(1)}_1 = \begin{bmatrix}
-30 \\
20  \\
20
\end{bmatrix}
$$

The hypothesis for the AND function, using a sigmoid activation function $g$, is then:

$$
h_{\Theta}(x) = g(-30 \cdot 1 + 20 \cdot x_1 + 20 \cdot x_2)
$$

The sigmoid function $g(z)$ maps any real number to the $(0, 1)$ interval, effectively acting as an activation function for the neuron.

![Sigmoid Function Graph](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/sigmoid.png)

The following single-unit AND model is trained by online gradient descent on one half of squared prediction error. That objective explains the extra sigmoid derivative in its update. Bernoulli cross-entropy would instead have logit derivative equal to prediction minus target.

```python
import numpy as np

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Sigmoid derivative function
def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Initialize weights and bias
np.random.seed(0)  # Seed for reproducibility
weights = np.random.randn(2)
bias = np.random.randn()

# Training data for AND gate
inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
targets = np.array([0, 0, 0, 1])  # Expected outputs for AND gate

# Hyperparameters
learning_rate = 0.1
epochs = 10000

# Training the neural network
for epoch in range(epochs):
    for x, y in zip(inputs, targets):
        # Forward pass
        z = np.dot(weights, x) + bias
        prediction = sigmoid(z)

        # Compute the error
        error = prediction - y

        # Backward pass (gradient descent)
        weights -= learning_rate * error * sigmoid_derivative(z) * x
        bias -= learning_rate * error * sigmoid_derivative(z)

# Function to compute the output of the AND gate neural network
def and_gate(x1, x2):
    x = np.array([x1, x2])
    z = np.dot(weights, x) + bias
    return sigmoid(z)

# Test the AND gate with all possible inputs
print("AND Gate Neural Network with Sigmoid Activation Function and Training")
print("Inputs    Output")
for x in inputs:
    output = and_gate(x[0], x[1])
    print(f"{x} -> {output:.4f} -> {round(output)}")

# Print final weights and bias
print("\nFinal weights:", weights)
print("Final bias:", bias)
```

- The implementation of an AND gate using a neural network with the sigmoid activation function involves several key steps, including defining the sigmoid function, initializing weights and bias, and training the model using gradient descent.
- The sigmoid function is used as the activation function in the neural network, and it is defined as $\sigma(z) = \frac{1}{1 + e^{-z}}$. This function outputs values between $0$ and $1$, making it suitable for binary classification tasks like the AND gate.
- The weights and bias are initially set to random values to start the training process. These values are crucial as they determine the output of the neuron based on the input features.
- The training data for the AND gate consists of all possible combinations of inputs ($00$, $01$, $10$, $11$) and their corresponding target outputs ($0$, $0$, $0$, $1$). This data is used to train the neural network.
- Hyperparameters such as the learning rate and the number of epochs are set to control the training process. The learning rate determines the step size for weight updates, while the number of epochs defines how many times the training loop will run.
- During the training loop, the model iterates over each input-target pair for a specified number of epochs. For each pair, the weighted sum (`z`) is computed, and the sigmoid function is applied to obtain the prediction.
- The error is calculated as the difference between the prediction and the target. This error is used to update the weights and bias using gradient descent. Specifically, the weights are adjusted by subtracting the product of the learning rate, the error, the derivative of the sigmoid function, and the input features. The bias is updated similarly.
- After training, the model's parameters (weights and bias) are optimized to correctly classify the AND gate inputs. A function is defined to compute the output of the neural network based on the trained weights and bias.
- The model is then tested with all possible input combinations for the AND gate ($00$, $01$, $10$, $11$). The output is printed for each input combination, showing both the raw sigmoid output and the rounded output to simulate the binary nature of the AND gate.
- The final weights and bias are printed to show the learned parameters of the model.

Running the seeded AND code gives the following output, with its probabilities rounded to four decimal places:

```
AND Gate Neural Network with Sigmoid Activation Function and Training
Inputs    Output
[0 0] -> 0.0002 -> 0
[0 1] -> 0.0555 -> 0
[1 0] -> 0.0555 -> 0
[1 1] -> 0.9340 -> 1

Final weights: [5.48338856 5.48302522]
Final bias: -8.316815762379743
```

### XNOR Requires a Nonlinear Hidden Representation

XNOR has target one at $(0,0)$ and $(1,1)$ and zero at $(0,1)$ and $(1,0)$. A single sigmoid followed by a threshold gives a linear boundary, so it cannot represent this arrangement.

To prove that limitation, suppose its score is $b+w_1x_1+w_2x_2$. The positive points require $b\geq0$ and $b+w_1+w_2\geq0$. The negative points require $b+w_1<0$ and $b+w_2<0$. Adding the negative inequalities gives $2b+w_1+w_2<0$, while adding the positive inequalities gives the same expression at least zero. No coefficients satisfy both.

![XNOR Function Graphical Representation](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/xnor.png)

Use two hidden sigmoid units instead:

$$
a_{\mathrm{OR}}=\sigma(20x_1+20x_2-10),\qquad
a_{\mathrm{AND}}=\sigma(20x_1+20x_2-30).
$$

Their activations approximately represent OR and AND. The XOR output is $\sigma(20a_{\mathrm{OR}}-40a_{\mathrm{AND}}-10)$, and XNOR is its complement. At a mixed input, OR is almost one and AND almost zero, so XOR's logit is almost ten. At $(1,1)$ both hidden units are near one, giving an XOR logit near minus thirty.

```python
import numpy as np
from scipy.special import expit
X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
W1 = np.array([[20.0, 20.0], [20.0, 20.0]])
b1 = np.array([-10.0, -30.0])
hidden = expit(X @ W1 + b1)
xor_probability = expit(hidden @ np.array([20.0, -40.0]) - 10.0)
xnor_probability = 1.0 - xor_probability
and_probability = hidden[:, 1]
np.testing.assert_array_equal(xor_probability >= 0.5, [0, 1, 1, 0])
np.testing.assert_array_equal(xnor_probability >= 0.5, [1, 0, 0, 1])
np.testing.assert_array_equal(and_probability >= 0.5, [0, 0, 0, 1])
assert hidden.shape == (4, 2)
print('hidden activations:\n', np.round(hidden, 6))
print('XOR:', np.round(xor_probability, 6))
print('XNOR:', np.round(xnor_probability, 6))
```

All four rows are asserted against the truth tables. Rounded XNOR outputs are approximately $0.999955,0.000046,0.000046,1.000000$. Finite sigmoid outputs lie strictly between zero and one; the last displayed value is rounded. These are constructed coefficients demonstrating representational capacity, not coefficients claimed to have been learned by a one-unit model.

The hidden layer has transformed the inputs so that a linear combination of its activations can separate the desired classes. A network can learn such features through its objective, but training is an additional optimization problem. A successful hand construction does not establish that every initialization will learn the same features.

## Keep Matrix Orientation and Parameter Counts Explicit

For a batch X of shape $(m,n)$ and a hidden layer with h units, use a weight matrix W of shape $(n,h)$ and bias vector b of shape $(h,)$. Then $Z=XW+b$ and $A=g(Z)$ both have shape $(m,h)$. NumPy broadcasts b across rows, adding the same learned unit biases to each example.

The course's column-vector convention instead writes a matrix $\Theta$ of shape $(h,n+1)$, with biases in its first column. The two conventions agree when $\Theta=[b\mid W^\top]$. Mixing their orientations can produce incorrect matrix products even when each individual formula looks familiar.

A two-input, two-hidden-unit, one-output network has $(2\cdot2+2)+(2\cdot1+1)=9$ parameters. The two-input, three-hidden-unit network in the learning example has 13. Counting biases separately helps check packing, unpacking, and gradient shapes.

## Nonlinear Activations Supply the Additional Capacity

Two linear layers compose to another linear layer:

$$
(XW_1+b_1)W_2+b_2=X(W_1W_2)+(b_1W_2+b_2).
$$

Adding linear layers alone therefore cannot solve the XNOR separation problem. A nonlinear activation changes the represented function class. Saturating sigmoid units can implement nearly Boolean features, while ReLU units represent piecewise-linear transformations.

Output activation also follows the target. A sigmoid models one binary probability. Several independent sigmoids can model multiple simultaneous labels without forcing their probabilities to sum to one. Softmax models mutually exclusive alternatives, with $p_k=e^{z_k}/\sum_j e^{z_j}$. Numerically, subtracting the maximum logit before exponentiation preserves those probabilities and prevents overflow.

For example, logits $(1,2,3)$ give probabilities approximately $(0.090031,0.244728,0.665241)$. Three independent sigmoids on the same logits give a different set of numbers whose sum exceeds one. The two output constructions correspond to different probability models and losses.

## Architecture Encodes How Parameters Are Shared

A 50-by-50 RGB image has 7,500 input values. A fully connected layer of 100 units contains $7{,}500(100)+100=750{,}100$ parameters. A convolution with sixteen 3-by-3 filters over three input channels has $3(3)(3)(16)+16=448$ learned parameters, reused at multiple spatial positions. These layers produce different representations; the counts illustrate parameter sharing rather than an accuracy comparison.

The biological diagrams motivate terminology, but the implemented model is defined by its operations, parameter shapes, and objective. Evaluating those explicit choices is more informative than attributing a network's behavior to a general resemblance to a brain.

## Hidden-Unit Order Does Not Identify a Unique Representation

The two hidden units in the logical example can be swapped if the output weights are swapped with them. Let P be a permutation matrix. Replacing $W_1$ by $W_1P$, the hidden bias row by $b_1P$, and the output vector by $P^\top w_2$ gives hidden activations $AP$. Since $PP^\top=I$,

$$
(AP)(P^\top w_2)=Aw_2.
$$

The complete output is unchanged. Using the parameters and input rows from the earlier truth-table example verifies this directly:

```python
import numpy as np
from scipy.special import expit

P = np.array([[0., 1.], [1., 0.]])
output_weights = np.array([20., -40.])
reordered_hidden = expit(X @ (W1 @ P) + b1 @ P)
reordered_output = expit(reordered_hidden @ (P.T @ output_weights) - 10.)
np.testing.assert_allclose(reordered_output, xor_probability, atol=1e-12)
```

The first hidden coordinate has no intrinsic claim to be the OR feature. A different learned ordering can represent the same function, so comparing individual weight entries from two fits can falsely suggest disagreement. Compare predictions or account for the permitted transformation before interpreting parameter differences.

The four logical inputs constrain the function on those four points. They do not establish its behavior on fractional inputs between them, and a large constructed sigmoid value is not an empirical calibration measurement. For images, generalization similarly depends on which variations are represented by training and evaluation inputs.

This separation between representational capacity, parameter identification, optimization, and predictive evaluation keeps the example precise. The truth-table assertions establish the computed logical decisions. The gradient checks establish the derivative calculation. A broader prediction claim requires observations from the intended input population.

## Reference

These notes are based on the free video lectures offered by Stanford University, led by Professor Andrew Ng. These lectures are part of the renowned Machine Learning course available on Coursera. For more information and to access the full course, visit the [Coursera course page](https://www.coursera.org/learn/machine-learning).
