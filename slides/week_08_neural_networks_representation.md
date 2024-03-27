## Neural Networks - Representation

Neural networks are a type of machine learning model that is designed to mimic the way the brain works. They are particularly useful for handling large and complex data sets that may have many features. Logistic regression, a common method for analyzing such data, can become computationally expensive when dealing with a large number of features. To avoid this issue, a subset of the features must be used, which can reduce the accuracy of the analysis. Neural networks have become more popular in recent years due to the increasing computational power of computers, which has made it possible to build large-scale neural networks. These networks are made up of neurons, which are connected by input and output wires and communicate through electric spikes. In an artificial neural network, input is fed through input wires and processed before being sent out through output wires. The input layer is the first layer, the output layer is the last, and any layers in between are called hidden layers. The weights of the model are represented by the $\theta$ vector, and the bias unit is represented by $x_0$. Neural networks are particularly useful in fields such as computer vision, where the data sets can be very large.

### Why do we need neural networks?

* We can have complex data sets with many (1000's) important features.
* Using logistic regression becomes expensive really fast.
* The only way to get around this is to use a subset of features. This, however, may result in less accuracy.

![many_features_classifier](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/many_features_classifier.png)

## Problems where n is large - computer vision

* Computer vision sees a matrix of pixel intensity values.
* To build a car detector: Build a training set of cars and not cars. Then test against a car.
* Plot two pixels (two pixel locations) and car or not car on the graph.

![cars](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/cars.png)


Feature space
## Problems where n is large - computer vision

* If we used 50 x 50 pixels $-->$ 2500 pixels, so n = 2500.
* If RGB then 7500.
* If 100 x 100 RB then $-->$ 50 000 000 features.
* Way too big.


### Neurons and the brain


* The desire to construct computers that mimicked brain activities drove the creation of neural networks (NNs).
* Used a lot in the 80s. Popularity diminished in 90s.
* Large-scale neural networks have just recently become computationally feasible due to the computational cost of NNs.


## Brain


* Hypothesis is that the brain has a single learning algorithm.
* If the optic nerve is rerouted to the auditory cortex, the auditory cortex is learns to see.
* If you rewrite the optic nerve to the somatosensory cortex then it learns to see.


### Model representation I

Neuron:

* Cell body
* Number of input wires (dendrites)
* Output wire (axon)

![neuron](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/neuron.png)

* Neurone gets one or more inputs through dendrites.
* Does processing.
* The output is sent along the axon.
* Neurons communicate through electric spikes.


## Artificial neural network - representation of a neurone


* Feed input via input wires.
* Logistic unit does computation.
* Sends output down output wires.

![neuron](https://user-images.githubusercontent.com/37275728/201517992-cdc14304-2af9-4821-bcae-71caa1a62663.png)


$$
x = \begin{bmatrix}
      x_{0} \\
      x_{1} \\
      x_2   \\
      x_3
    \end{bmatrix}
$$


$$
    \theta  = \begin{bmatrix}
      \theta_{0} \\
      \theta_{1} \\
      \theta_2   \\
      \theta_3
    \end{bmatrix}
$$

* The diagram above represents a single neurone.
* $x_0$ is called the bias unit.
* $\theta$ vector is called the weights of a model.

![hidden_layer](https://user-images.githubusercontent.com/37275728/201517995-ff2af22c-ea22-4be9-9bfc-b7e6c771d69c.png)

* First layer is the input layer.
* Final layer is the output layer - produces value computed by a hypothesis.
* Middle layer(s) are called the hidden layers.
* You don't observe the values processed in the hidden layer.
* Every input/activation goes to every node in following layer.


$$a^{(2)}_1 = g(\Theta^{(1)}_{10}x_0+\Theta^{(1)}_{11}x_1+\Theta^{(1)}_{12}x_2+\Theta^{(1)}_{13}x_3)$$
$$a^{(2)}_2 = g(\Theta^{(1)}_{20}x_0+\Theta^{(1)}_{21}x_1+\Theta^{(1)}_{22}x_2+\Theta^{(1)}_{23}x_3)$$
$$a^{(2)}_1 = g(\Theta^{(1)}_{30}x_0+\Theta^{(1)}_{31}x_1+\Theta^{(1)}_{32}x_2+\Theta^{(1)}_{33}x_3)$$
$$h_{\Theta}(x) = g(\Theta^{(2)}_{10}a^{(2)}_0+\Theta^{(2)}_{11}a^{(2)}_1+\Theta^{(2)}_{12}a^{(2)}_2+\Theta^{(2)}_{13}a^{(2)}_3)$$

### Model representation II
In this section, we'll look at how to do the computation efficiently using a vectorized approach. We'll also look at why NNs are useful and how we can use them to learn complicated nonlinear things.


Let's define a few more terms:

$$z^{(2)}_1 = \Theta^{(1)}_{10}x_0+\Theta^{(1)}_{11}x_1+\Theta^{(1)}_{12}x_2+\Theta^{(1)}_{13}x_3$$


We can now write:
$$a^{(2)}_1 = g(z^{(2)}_1)$$

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

* $z^{(2)}$ is a $3x1$ vector.
* $a^{(2)}$ ia also a $3x1$ vector.
* Middle layer(s) are called the hidden layers.
* $g()$ applies the sigmoid (logistic) function element wise to each member of the $z^{(2)}$ vector.
* Obviously the "activation" for the input layer is just the input!
* $a^{(1)}$ is the vector of inputs.
* $a^{(2)}$ is the vector of values calculated by the $g(z^{(2)})$ function.
* We send our input values to the hidden layers and let them learn which values produce the best final result to feed into the final output layer.
* This process is also called forward propagation.

Other architectural designs are also possible:

* More/less nodes per layer.
* More layers.


![multi_layer](https://user-images.githubusercontent.com/37275728/201517998-e5f9f245-a6f1-4aed-8a58-fcb0178f38c4.png)


Layer 2 has three hidden units (plus bias), layer 3 has two hidden units (plus bias), and by the time you get to the output layer, you have a really intriguing non-linear hypothesis.

### AND function

![and_function](https://user-images.githubusercontent.com/37275728/201518002-72b41fb7-ca3f-4612-aa65-c34f58138737.png)

Let $x_0 = 1$ and theta vector be:

$$
  \Theta^{(1)}_1 = \begin{bmatrix}
    -30 \\
    20  \\
    20
  \end{bmatrix}
$$

Then hypothesis is:

$$h_{\Theta}(x) = g(-30 \cdot 1 + 20 \cdot x_1 + 20 \cdot x_2)$$

![sigmoid](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/sigmoid.png)

### XNOR function

![xnor](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/xnor.png)