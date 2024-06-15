## Anomaly Detection in Machine Learning

Anomaly detection involves identifying data points that significantly differ from the majority of the data, often signaling unusual or suspicious activities. This technique is widely used across various domains, such as fraud detection, manufacturing, and system monitoring.

### Concept of Anomaly Detection

- **Baseline Establishment**: The first step in anomaly detection is to establish a normal pattern or baseline from the dataset.
- **Probability Threshold (Epsilon)**: Data points are flagged as anomalies if their probability, according to the established model, falls below a threshold $\epsilon$.
- If $p(x_{\text{test}}) < \epsilon$, the data point is considered an anomaly.
- If $p(x_{\text{test}}) \geq \epsilon$, the data point is considered normal.
- **Threshold Selection**: The value of $\epsilon$ is critical and is chosen based on the desired confidence level and the specific context of the application.

![Illustration of Anomaly Detection](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/anomaly.png)

### Applications of Anomaly Detection

I. Fraud Detection:

- User behavior metrics (online time, login location, spending patterns) are analyzed.
- A model of typical user behavior is created, and deviations are flagged as potential fraud.

II. Manufacturing: In scenarios like aircraft engine production, anomalies can indicate defects or potential failures.

III. Data Center Monitoring: Monitoring metrics (memory usage, disk accesses, CPU load) to identify machines that are likely to fail.

### Utilizing the Gaussian Distribution

- **Mean ($\mu$) and Variance ($\sigma^2$)**: The Gaussian distribution is characterized by these parameters.
- **Probability Calculation**:
  
$$
p(x; \mu; \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$
  
![Gaussian Distribution](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/gaussian.png)

- **Data Fitting**: Estimating the Gaussian distribution from the dataset to represent the normal behavior.

![Data Fitting with Gaussian](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/data_fit.png)

### Evaluating Anomaly Detection Systems

- **Data Split**: Divide the data into training, cross-validation, and test sets.
- **Performance Metrics**: Precision and recall are often used to evaluate the effectiveness of the anomaly detection system.
- **Fine-Tuning**: Adjust the model and $\epsilon$ based on the performance metrics to improve accuracy.

### Algorithm Steps

1. **Feature Selection**: Choose features $x_i$ that might be indicators of anomalous behavior.

2. **Parameter Fitting**: Calculate the mean ($\mu_j$) and variance ($\sigma_j^2$) for each feature
     
$$
\mu_j = \frac{1}{m} \sum_{i=1}^m x_j^{(i)}
$$

$$
\sigma_j^2 = \frac{1}{m} \sum_{i=1}^m (x_j^{(i)} - \mu_j)^2
$$

3. **Probability Computation for New Example**: For a new example $x$, compute the probability $p(x)$:

$$
p(x) = \prod_{j=1}^n \frac{1}{\sqrt{2\pi\sigma_j^2}} \exp\left(-\frac{(x_j - \mu_j)^2}{2\sigma_j^2}\right)
$$

### Developing and Evaluating an Anomaly Detection System

I. **Labeled Data**: Have a dataset where $y=0$ indicates normal (non-anomalous) examples, and $y=1$ represents anomalous examples.

II. **Data Division**: Separate the dataset into a training set (normal examples), a cross-validation (CV) set, and a test set, with both the CV and test sets including some anomalous examples.

III. **Example Case**:

- Imagine a dataset with 10,000 normal (good) engines and 50 flawed (anomalous) engines.
- Training set: 6,000 good engines.
- CV set: 2,000 good engines, 10 anomalous.
- Test set: 2,000 good engines, 10 anomalous.

IV. **Evaluation Metrics**:

- True positives (TP), false positives (FP), false negatives (FN), and true negatives (TN).
- Precision (the proportion of true positives among all positives identified by the model).
- Recall (the proportion of true positives identified out of all actual positives).
- F1-score (a harmonic mean of precision and recall, providing a balance between them).

### Selecting a Good Evaluation Metric

- **F1-Score**: This is particularly useful in the context of anomaly detection where the class distribution is highly imbalanced (many more normal than anomalous examples).
- **Precision and Recall**: These metrics provide a more nuanced understanding of the system’s performance, especially when the cost of false positives and false negatives varies.

### Concept of Multivariate Gaussian Distribution

- **Scenario**: Imagine fitting a Gaussian distribution to two features, such as CPU load and memory usage.
- **Example**: In a test set, consider an example with a high memory use (x1 = 1.5) but low CPU load (x2 = 0.4). Individually, these values might fall within normal ranges, but their combination could be anomalous.

![Example of Multivariate Gaussian Distribution](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/mult_gauss.png)

### Parameters of the Multivariate Gaussian Model

- **Mean Vector ($\mu$)**: An n-dimensional vector representing the mean of each feature.
- **Covariance Matrix ($\Sigma$)**: An $[n \times n]$ matrix, capturing how each pair of features varies together.
- **Probability Density Function**:

$$
p(x; \mu; \Sigma) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu)^T \Sigma^{-1}(x - \mu)\right)
$$
  
![Covariance Matrix](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/cov_matrix_sigma.png)

### Gaussian Model vs. Multivariate Gaussian Model

#### Gaussian Model

- **Usage**: More commonly used in anomaly detection.
- **Feature Creation**: Requires manual creation of features to capture unusual combinations in values.
- **Computational Efficiency**: Generally more computationally efficient.
- **Scalability**: Scales well to large feature vectors.
- **Training Set Size**: Works effectively even with small training sets.

#### Multivariate Gaussian Model

- **Usage**: Used less frequently.
- **Feature Correlations**: Directly captures correlations between features without needing extra feature engineering.
- **Computational Cost**: Less efficient computationally.
- **Data Requirements**: Requires more examples than the number of features (m > n).
- **Advantage**: Can detect anomalies that occur due to unusual combinations of normal-appearing individual features.
