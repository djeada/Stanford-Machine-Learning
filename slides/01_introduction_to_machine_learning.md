# Introduction to Machine Learning
Machine Learning (ML), a subset of artificial intelligence, is the scientific study of algorithms and statistical models that computer systems use to effectively perform a specific task without using explicit instructions. It relies on patterns and inference instead. ML algorithms build a mathematical model based on sample data, known as "training data," to make predictions or decisions without being explicitly programmed to perform the task. 

## Types of Machine Learning
1. **Supervised Learning**: This involves learning a function that maps an input to an output based on example input-output pairs. It infers a function from labeled training data consisting of a set of training examples.

2. **Unsupervised Learning**: Here, the algorithm learns from plain data without any corresponding output variables to guide the learning process.

3. **Reinforcement Learning**: This type involves algorithms that learn optimal actions through trial and error. This type of learning is concerned with how software agents ought to take actions in an environment so as to maximize some notion of cumulative reward.

## Deep Learning in Machine Learning
Deep learning, a subset of ML, uses layered neural networks to simulate human decision-making. Key aspects include:

- **Convolutional Neural Networks (CNNs)**: Used primarily for processing data with a grid-like topology, e.g., images.
- **Recurrent Neural Networks (RNNs)**: Suitable for processing sequential data, e.g., time series or text.

### Applications and Suitability
Deep learning is particularly useful in areas such as image and speech recognition, where the volume and complexity of data surpass the capabilities of traditional algorithms.

## Prevalence of Machine Learning
ML's growing importance is attributed to its ability to leverage large amounts of data, enabling machines to recognize patterns and make informed decisions. It's considered a key to building systems that can simulate human intelligence.

## Practical Applications of Machine Learning

- **Database Mining**: Utilized in areas like internet search, understanding user behavior, medical diagnosis, and genomics.
- **Autonomous Systems**: Self-driving cars, drones, and other systems that make decisions based on data.
- **Recognition Tasks**: Such as handwriting, speech, and facial recognition.
- **Natural Language Processing**: Enables machines to understand and respond to human language.
- **Self-Customizing Programs**: Used in personalized recommendation systems like those on Netflix, Amazon, and iTunes.

## Foundational Concepts and Definitions

- **Arthur Samuel (1959)**: Defined ML as a field of study that enables computers to learn without being explicitly programmed.
- **Tom Mitchell (1997)**: Proposed a formal definition - a computer program is said to learn from experience E concerning task T and performance measure P if its performance on T, as measured by P, improves with experience E.

## Supervised Learning
Supervised learning, a dominant approach in machine learning, focuses on using labeled datasets to train algorithms. These algorithms are then applied to classify data or predict outcomes with a high level of accuracy.

### House Price Prediction - Regression Example
Consider a typical problem: predicting house prices based on their size. For instance, estimating the price of a 750 square feet house. This problem can be approached using different models:

- Linear Regression: Using a straight line to fit the data.
- Polynomial Regression: Fitting a curved line (e.g., second-order polynomial).

![house_price_prediction](https://user-images.githubusercontent.com/37275728/201469371-44d7837e-be00-4328-a978-0a63783e08c1.png)

### Tumor Size and Prognosis - Classification Example
In a classification problem, the aim is to categorize data into discrete groups. For example, determining if a tumor is malignant based on its size is a classic classification problem.

![malignant_cancer](https://user-images.githubusercontent.com/37275728/201469417-2f5449d2-62b4-4f36-8eb2-77abc4cb0adf.png)

### Combining Multiple Attributes
When considering multiple attributes, such as age and tumor size, classification becomes more intricate. We can use algorithms to draw boundaries between different classes in this multidimensional feature space.

![age_vs_tumor](https://user-images.githubusercontent.com/37275728/201469444-d574fbc0-8ed9-4378-a8a4-9c8f44062199.png)

## Unsupervised Learning
Unsupervised learning, in contrast to supervised learning, uses datasets without predefined labels. The goal here is to identify patterns or structures within the data.

### Clustering Algorithm Examples

- **Google News**: Clustering similar news stories.
- **Genomics**: Classifying types of genetic expressions.
- **Microarray Data**: Grouping individuals based on gene expression.
- **Computer Clusters**: Organizing clusters to optimize performance or identify faults.
- **Social Network Analysis**: Analyzing customer data in social networks.
- **Astronomical Data**: Classifying celestial objects based on their properties.

### The Cocktail Party Problem

The Cocktail Party Problem is an audio processing challenge where the goal is to separate individual audio sources from a mixture of sounds. This scenario is common in environments like parties, where multiple people are talking simultaneously, and you want to isolate the speech of a single person from a recording that captures all the speakers.

- The objective in the Cocktail Party Problem is to extract distinct audio signals from a complex mixture. This mixture can be influenced by factors such as room acoustics and overlapping speech, making it a challenging task.

- A common approach to solve this problem is Blind Source Separation (BSS). The goal of BSS is to recover original audio signals (the sources) from mixed signals without knowledge about the mixing process.

- Singular Value Decomposition (SVD) is a crucial technique for tackling this issue. SVD decomposes a matrix, representing audio data, into three matrices that expose the fundamental structure of the data. The mathematical representation of SVD applied to a matrix \(X\) (representing mixed audio signals) is:

$$
X = U \Sigma V^*
$$

Where:

- $X$ is the matrix of mixed signals.
- $U$ and $V$ are orthogonal matrices.
- $S$ is a diagonal matrix with singular values.

SVD is particularly useful here because it provides a robust way to deconstruct the mixed signals into components that are easier to separate and analyze. The orthogonal matrices $U$ and $V$ represent the bases in the original and transformed spaces, respectively, while the singular values in $S$ represent the strength of each component in the data. By filtering or modifying these components, we can work towards isolating individual audio sources from the mix.

This mock code illustrates the use of SVD for signal reconstruction and ICA for source separation, addressing the Cocktail Party Problem in audio processing:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

# Function to perform Singular Value Decomposition
def perform_svd(X):
    U, S, V = np.linalg.svd(X, full_matrices=False)
    return U, S, V

# Function to plot individual signals
def plot_individual_signals(signals, title):
    plt.figure(figsize=(10, 4))
    for i, signal in enumerate(signals):
        plt.plot(signal, label=f'{title} {i+1}')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Function to plot a single signal
def plot_single_signal(signal, title):
    plt.figure(figsize=(10, 4))
    plt.plot(signal)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Generate original signals for comparison
def generate_original_signals():
    np.random.seed(0)
    time = np.linspace(0, 8, 4000)
    s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
    s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
    return np.c_[s1, s2]

# Generate mixed signals for demonstration purposes
def simulate_mixed_signals():
    S = generate_original_signals()
    S += 0.2 * np.random.normal(size=S.shape)  # Add noise
    S /= S.std(axis=0)  # Standardize data

    # Mixing matrix
    A = np.array([[1, 1], [0.5, 2]])
    X = np.dot(S, A.T)  # Generate observations
    return X, S  # Return both mixed signals and original sources

# Generate mixed and original signals
mixed_signals, original_sources = simulate_mixed_signals()

# Perform SVD on the mixed signals
U, S, V = perform_svd(mixed_signals)

# Reconstruct the signals using the top singular values
top_k = 2  # Number of top components to use for reconstruction
reconstructed = np.dot(U[:, :top_k], np.dot(np.diag(S[:top_k]), V[:top_k, :]))

# Apply Independent Component Analysis (ICA) to separate the sources
ica = FastICA(n_components=2)
sources = ica.fit_transform(reconstructed.T).T

# Sum the mixed signals to represent the original combined signal
summed_mixed_signals = mixed_signals.sum(axis=1)

# Plot the summed mixed signals, reconstructed signals, and separated sources
plot_single_signal(summed_mixed_signals, 'Summed Mixed Signals')
plot_individual_signals(reconstructed.T, 'Reconstructed Signals')
plot_individual_signals(sources, 'Separated Sources')
```

Original signal:

![output](https://github.com/djeada/Stanford-Machine-Learning/assets/37275728/2aad3b3d-d26a-4f41-bd2e-790019c5725d)

Reconstructed signals:

![output (1)](https://github.com/djeada/Stanford-Machine-Learning/assets/37275728/a9ebcc4f-2fda-4ef1-b0b9-a2460164fe13)

Separated sources:

![output (2)](https://github.com/djeada/Stanford-Machine-Learning/assets/37275728/c236f845-270c-4f78-aeb4-5f6a080ac71e)

## Reference

These notes are based on the free video lectures offered by Stanford University, led by Professor Andrew Ng. These lectures are part of the renowned Machine Learning course available on Coursera. For more information and to access the full course, visit the [Coursera course page](https://www.coursera.org/learn/machine-learning).

