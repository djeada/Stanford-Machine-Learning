## Large Scale Machine Learning

Training machine learning models on large datasets poses significant challenges due to the computational intensity involved. To effectively handle this, various techniques such as stochastic gradient descent and online learning are employed. Let's delve into these methods and understand how they facilitate large-scale learning.

### Learning with Large Datasets

To achieve high performance, one effective approach is using a low bias algorithm on a massive dataset. For example:

- **Scenario**: Consider a dataset with `m = 100,000,000` examples.
- **Challenge**: Training a model like logistic regression on this scale requires significant computation for each gradient descent step.
  
Logistic Regression Update Rule:

$$
\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^m(h_{\theta}(x^{(i)}) - y^{(i)}) x_j^{(i)}
$$

- **Approach**: Experiment with a smaller sample (say, 1000 examples) to test performance.

![Learning Curve Example](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/learning_curve.png)

Diagnosing Bias vs. Variance: 

- A large gap between training and cross-validation errors indicates high variance, suggesting more data might help.
- A small gap implies high bias, where additional data may not improve performance.

### Stochastic Gradient Descent (SGD)

SGD optimizes the learning process for large datasets by updating parameters more frequently:

- **Batch Gradient Descent**: Traditional method that sums gradient terms over all examples before updating parameters. It's inefficient for large datasets.

- **SGD Approach**: Randomly shuffle the training examples. Update $\theta_j$ for each example:

$$
\theta_j := \theta_j - \alpha (h_{\theta}(x^{(i)}) - y^{(i)}) x_j^{(i)}
$$

- This results in parameter updates for every training example, rather than at the end of a full pass over the data.

![Stochastic Gradient Descent Illustration](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/stochastic.png)

- **Convergence Observation**: Observe the cost function versus the number of iterations.

![SGD Convergence](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/stochastic_convergence.png)

### Online Learning

- **Concept**: Online learning is a dynamic form of learning where the model updates its parameters as new data arrives.
- **Use Case**: Ideal for continuously evolving datasets or when data comes in streams.
- **Advantage**: Allows the model to adapt to new trends or changes in the data over time.

### Example: Product Search on a Cellphone Website

Imagine a cellphone-selling website scenario:

- **User Queries**: Users enter queries like "Android phone 1080p camera."
- **Ranking Phones**: The objective is to present the user with a list of ten phones, ranked according to their relevance or appeal.
- **Feature Vectors**: Generate a feature vector (x) for each phone, tailored to the user’s specific query.
- **Click Prediction (CTR)**: Learn the probability $p(y = 1 | x; \theta)$, where $y = 1$ if a user clicks on a phone link, and $y = 0$ otherwise. This probability represents the predicted click-through rate (CTR).
- **Utilizing CTR**: Rank and display phones based on their estimated CTR, showing the most likely to be clicked options first.

### Map Reduce in Large Scale Machine Learning

Map Reduce is a programming model designed for processing large datasets across distributed clusters. It simplifies parallel computation on massive scales, making it a cornerstone for big data analytics and machine learning where traditional single-node processing falls short.

![Map Reduce Illustration](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/map_reduce.png)

### Understanding Map Reduce

Map Reduce works by breaking down the processing into two main phases: Map phase and Reduce phase.

I. Map Phase:

- **Operation**: This phase involves taking a large input and dividing it into smaller sub-problems. Each sub-problem is processed independently.
- **Function**: The map function applies a specific operation to each sub-problem. For example, it might involve filtering data or sorting it.
- **Output**: The result of the map function is a set of key-value pairs.

II. Reduce Phase:

- **Operation**: In this phase, the output from the map phase is combined or reduced into a smaller set of tuples.
- **Function**: The reduce function aggregates the results to form a consolidated output.
- **Examples**: Summation, counting, or averaging over large datasets.

### Implementation of Map Reduce

One popular framework for implementing Map Reduce is Apache Hadoop, which handles large scale data processing across distributed clusters.

Hadoop Ecosystem:
 
- **Hadoop Distributed File System (HDFS)**: Splits data into blocks and distributes them across the cluster.
- **MapReduce Engine**: Manages the processing by distributing tasks to nodes in the cluster, handling task failures, and managing communications.

Process Flow:

1. **Data Distribution**: Data is split and distributed across the HDFS.
2. **Job Submission**: A MapReduce job is defined and submitted to the cluster.
3. **Mapping**: The map tasks process the data in parallel.
4. **Shuffling**: Data is shuffled and sorted between the map and reduce phases.
5. **Reducing**: Reduce tasks aggregate the results.
6. **Output Collection**: The final output is assembled and stored back in HDFS.

### Advantages of Map Reduce

- **Scalability**: Can handle petabytes of data by distributing tasks across numerous machines.
- **Fault Tolerance**: Automatically handles failures. If a node fails, tasks are rerouted to other nodes.
- **Flexibility**: Capable of processing structured, semi-structured, and unstructured data.

### Applications in Machine Learning

- **Large-scale Data Processing**: For training models on datasets that are too large for a single machine, especially for tasks like clustering, classification, and pattern recognition.
- **Parallel Training**: Training multiple models or model parameters in parallel, reducing the overall training time.
- **Data Preprocessing**: Large-scale data cleaning, transformation, and feature extraction.

### Forms of Parallelization

- **Multiple Machines**: Distribute the data and computations across several machines.
- **Multiple CPUs**: Utilize several CPUs within a machine.
- **Multiple Cores**: Take advantage of multi-core processors for parallel processing.

### Local and Distributed Parallelization

- **Numerical Linear Algebra Libraries**: Some libraries can automatically parallelize computations across multiple cores.
- **Vectorization**: With an efficient vectorization implementation, local libraries might handle much of the optimization, reducing the need for manual parallelization.
- **Distributed Systems**: For data that's too large for a single system, distributed computing frameworks like Hadoop are used. These frameworks apply the Map Reduce paradigm to process data across multiple machines.

## Reference

These notes are based on the free video lectures offered by Stanford University, led by Professor Andrew Ng. These lectures are part of the renowned Machine Learning course available on Coursera. For more information and to access the full course, visit the [Coursera course page](https://www.coursera.org/learn/machine-learning).
