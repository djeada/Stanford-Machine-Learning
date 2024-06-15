## Neural Networks Introduction

Neural networks represent a cornerstone in the field of machine learning, drawing inspiration from neurological processes within the human brain. These networks excel in processing complex datasets with numerous features, transcending traditional methods like logistic regression in both scalability and efficiency. Particularly, logistic regression can become computationally intensive and less practical with a high-dimensional feature space, often necessitating the selection of a feature subset, which might compromise the model's accuracy.

A neural network comprises an intricate architecture of neurons, analogous to the brain's neural cells, linked through synaptic connections. These connections facilitate the transmission and processing of information across the network. The basic structure of a neural network includes:

- **Input Layer:** Serves as the entry point for data into the network.
- **Hidden Layers:** Intermediate layers that perform computations and feature transformations.
- **Output Layer:** Produces the final output of the network.

Each neuron in these layers applies a weighted sum to its inputs, followed by a nonlinear activation function. The weights of these connections are represented by the matrix $\theta$, while the bias term is denoted as $x_0$. These parameters are learned during the training process, enabling the network to capture complex patterns and relationships within the data.

### Mathematical Representation

Consider a neural network with $L$ layers, each layer $l$ having $s_l$ units (excluding the bias unit). The activation of unit $i$ in layer $l$ is denoted as $a_i^{(l)}$. The activation function applied at each neuron is usually a nonlinear function like the sigmoid or ReLU function. The cost function for a neural network, often a function like cross-entropy or mean squared error, is minimized during the training process to adjust the weights $\theta$.

### Significance in Computer Vision

Neural networks particularly shine in domains like computer vision, where data often involves high-dimensional input spaces. For instance, an image with a resolution of 50x50 pixels, considering only grayscale values, translates to 2,500 features. If we incorporate color channels (RGB), the feature space expands to 7,500 dimensions. Such high-dimensional data is unmanageable for traditional algorithms but is aptly handled by neural networks through feature learning and dimensionality reduction techniques.

### Neuroscience Inspiration

The inception of neural networks was heavily influenced by the desire to replicate the brain's learning mechanism. This fascination led to the development and evolution of various neural network architectures through the decades. A notable hypothesis in neuroscience suggests that the brain might utilize a universal learning algorithm, adaptable across different sensory inputs and functions. This adaptability is exemplified in experiments where rerouting sensory nerves (e.g., optic to auditory) results in the corresponding cortical area adapting to process the new form of input, a concept that echoes in the design of flexible and adaptive neural networks.

### Model Representation I

#### Neuron Model in Biology

In biological terms, a neuron consists of three main parts:

- **Cell Body:** Central part of a neuron where inputs are aggregated.
- **Dendrites:** Input wires which receive signals from other neurons.
- **Axon:** Output wire that transmits signals to other neurons.

Biological neurons process signals through a combination of electrical and chemical means, sending impulses (or spikes) along the axon as a response to input stimuli received through dendrites.

![Biological Neuron Diagram](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/neuron.png)

#### Artificial Neural Network: Neuron Representation

In artificial neural networks, the neuron or 'node' functions similarly to its biological counterpart but in a simplified and abstracted manner:

- **Inputs:** Received through 'input wires', akin to dendrites.
- **Computation:** Performed by a logistic unit within the neuron.
- **Output:** Sent down 'output wires', similar to the axon.

Each neuron in an artificial neural network processes its inputs using a weighted sum and an activation function. The result is then passed on to subsequent neurons or to the output layer.

![Artificial Neuron Diagram](https://user-images.githubusercontent.com/37275728/201517992-cdc14304-2af9-4821-bcae-71caa1a62663.png)

#### Mathematical Model of a Neuron

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

#### Layers in a Neural Network

- **Input Layer:** The initial layer that receives input data.
- **Hidden Layers:** Intermediate layers where data transformations occur. The activations within these layers are not directly observable.
- **Output Layer:** Produces the final output based on the computations performed by the network.

The connectivity pattern in a neural network typically involves each neuron in one layer being connected to all neurons in the subsequent layer.

![Hidden Layer Representation](https://user-images.githubusercontent.com/37275728/201517995-ff2af22c-ea22-4be9-9bfc-b7e6c771d69c.png)

#### Activation and Output Computation

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

### Model Representation in Neural Networks II

Neural networks process large amounts of data, necessitating efficient computation methods. Vectorization is a key technique used to achieve this efficiency. It allows for the simultaneous computation of multiple operations, significantly speeding up the training and inference processes in neural networks.

#### Defining Vectorized Terms

To illustrate vectorization, consider the computation of the activation for neurons in a layer. The activation of the $i^{th}$ neuron in layer 2, $a^{(2)}_i$, is based on a linear combination of inputs followed by a nonlinear activation function $g$ (e.g., sigmoid function). This can be represented as:

$$z^{(2)}_i = \Theta^{(1)}_{i0}x_0+\Theta^{(1)}_{i1}x_1+\Theta^{(1)}_{i2}x_2+\Theta^{(1)}_{i3}x_3$$

Hence, the activation $a^{(2)}_i$ is given by:

$$a^{(2)}_i = g(z^{(2)}_i)$$

#### Vector Representation

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

#### Architectural Flexibility

Neural network architectures can vary in complexity:

- **Variation in Node Count:** The number of neurons in each layer can be adjusted based on the complexity of the task.
- **Number of Layers:** Additional layers can be added to create deeper networks, which are capable of learning more complex patterns.

![Multi-Layer Neural Network](https://user-images.githubusercontent.com/37275728/201517998-e5f9f245-a6f1-4aed-8a58-fcb0178f38c4.png)

In the above example, layer 2 has three hidden units, and layer 3 has two hidden units. By adjusting the number of layers and nodes, neural networks can model complex nonlinear hypotheses, enabling them to tackle a wide range of problems from simple linear classification to complex tasks in computer vision and natural language processing.

### Neural Networks for Logical Functions: AND and XNOR

#### The AND Function

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

#### The XNOR Function

The XNOR (exclusive-NOR) function is another logical operation that outputs true if both inputs are either true or false.

![XNOR Function Graphical Representation](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/xnor.png)

Unlike the AND function, constructing an XNOR function requires more than one neuron because it is a non-linear function. A neural network with at least one hidden layer containing multiple neurons is necessary to model the XNOR function. The network would typically combine basic logical functions like AND, OR, and NOT in its layers to replicate the XNOR behavior.

In these examples, the neural network uses a weighted combination of inputs to activate a neuron. The weights (in $\Theta$) and bias terms determine how the neuron responds to different input combinations. For binary classification tasks like AND and XNOR, the sigmoid function works well because it outputs values close to 0 or 1, analogous to the binary nature of these logical operations.

## Reference

These notes are based on the free video lectures offered by Stanford University, led by Professor Andrew Ng. These lectures are part of the renowned Machine Learning course available on Coursera. For more information and to access the full course, visit the [Coursera course page](https://www.coursera.org/learn/machine-learning).

