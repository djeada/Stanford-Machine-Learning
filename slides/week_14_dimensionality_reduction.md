## Dimensionality Reduction with Principal Component Analysis (PCA)

Principal Component Analysis (PCA) is a widely used technique in machine learning for dimensionality reduction. It simplifies the complexity in high-dimensional data while retaining trends and patterns.

### Understanding PCA

- **Objective**: PCA seeks to find a lower-dimensional surface onto which data points can be projected with minimal loss of information.
- **Methodology**: It involves computing the covariance matrix of the data, finding its eigenvectors (principal components), and projecting the data onto a space spanned by these eigenvectors.
- **Applications**: PCA is used for tasks such as data compression, noise reduction, visualization, feature selection, and enhancing the performance of machine learning algorithms.

### Compression with PCA

- **Benefits**:
  - Speeds up learning algorithms.
  - Saves storage space.
  - Focuses on the most relevant features, discarding less important ones.
- **Example**: Different units of the same attribute can be reduced to a single, more representative dimension.

  ![Example of Dimension Reduction](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/compression_units.png)

### Visualization through PCA

- **Challenge**: High-dimensional data is difficult to visualize.
- **Solution**: PCA can reduce dimensions to make data visualization more feasible and interpretable.
- **Example**: Representing 50 features of a dataset as a 2D plot, simplifying analysis and interpretation.

  ![Example of Data Table](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/table.png)

### PCA Problem Formulation

1. **Goal**: To find a lower-dimensional representation that minimizes the projection error (distance between original and projected points).
2. **Projection Error**: PCA aims to minimize the sum of the squares of these distances.

   ![Visualizing PCA Projection](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/pca.png)

3. **PCA vs. Linear Regression**:
   - Linear Regression: Minimizes the vertical distances between data points and the fitted line (predictive model).
   - PCA: Minimizes the orthogonal distances to the line (data representation model), without distinguishing between dependent and independent variables.

### Selecting Principal Components

- **Number of Components**: The choice of how many principal components to retain depends on the trade-off between minimizing projection error and reducing dimensionality.
- **Variance Retained**: Ideally, the selected components should retain most of the variance of the original data.

### Principal Component Analysis (PCA) Algorithm

Principal Component Analysis (PCA) is a systematic process for reducing the dimensionality of data. Here's a breakdown of the PCA algorithm:

1. **Covariance Matrix Computation**:
   
   Calculate the covariance matrix $\Sigma$:

$$\Sigma = \frac{1}{m} \sum_{i=1}^{n} (x^{(i)})(x^{(i)})^T$$
   
   Here, $\Sigma$ is an $[n \times n]$ matrix, with each $x^{(i)}$ being an $[n \times 1]$ matrix.

2. **Eigenvector Calculation**:
   
   Compute the eigenvectors of the covariance matrix $\Sigma$:

$$
[U,S,V] = \text{svd}(\Sigma)
$$
   
   The matrix $U$ will also be an $[n \times n]$ matrix, with its columns being the eigenvectors we seek.

3. **Choosing Principal Components**:
   
   Select the first $k$ eigenvectors from $U$, this is $U_{\text{reduce}}$.

4. **Calculating Compressed Representation**:
   
   For each data point $x$, compute its new representation $z$:
 
$$z = (U_{\text{reduce}})^T \cdot x$$

### Reconstruction from Compressed Representation

Is it possible to go back from a lower dimension to a higher one? While exact reconstruction is not possible (since some information is lost), an approximation can be obtained:

$$x_{\text{approx}} = U_{\text{reduce}} \cdot z$$

This approximates the original data in the higher-dimensional space but aligned along the principal components.

### Choosing the Number of Principal Components

The number of principal components ($k$) is a crucial choice in PCA. The objective is to minimize the average squared projection error while retaining most of the variance in the data:

- **Average Squared Projection Error**:

$$\frac{1}{m} \sum_{i=1}^{m} ||x^{(i)} - x_{\text{approx}}^{(i)}||^2$$

- **Total Data Variation**:

$$\frac{1}{m} \sum_{i=1}^{m} ||x^{(i)}||^2$$

- **Choosing $k$**:
  The fraction of variance retained is often set as a threshold (e.g., 99%):

$$
\frac{\frac{1}{m} \sum_{i=1}^{m} ||x^{(i)} - x_{\text{approx}}^{(i)}||^2}
{\frac{1}{m} \sum_{i=1}^{m} ||x^{(i)}||^2} 
\leq 0.01
$$

### Applications of PCA

1. **Compression**: Reducing data size for storage or faster processing.
2. **Visualization**: With $k=2$ or $k=3$, data can be visualized in 2D or 3D space.
3. **Limitation**: PCA should not be used indiscriminately to prevent overfitting. It removes data dimensions without understanding their importance.
4. **Usage Advice**: It's recommended to try understanding the data without PCA first and apply PCA if it is believed to aid in achieving specific objectives.
