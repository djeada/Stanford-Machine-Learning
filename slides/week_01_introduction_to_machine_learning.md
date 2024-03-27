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
