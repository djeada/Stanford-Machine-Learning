## Clustering in Unsupervised Learning

Unsupervised learning, a core component of machine learning, focuses on discerning the inherent structure of data without any labeled examples. Clustering, a pivotal task in unsupervised learning, aims to organize data into meaningful groups or clusters. A quintessential algorithm for clustering is the K-means algorithm.

### Overview of Clustering

I. **Objective**: To uncover hidden patterns and structures in unlabeled data.

II. **Applications**:

- Market segmentation, categorizing clients based on varying characteristics.
- Social network analysis to understand relationship patterns.
- Organizing computer clusters and data centers based on network structure and location.
- Analyzing astronomical data for insights into galaxy formation.

### The K-means Algorithm

K-means is a widely-used algorithm for clustering, celebrated for its simplicity and efficacy. It works as follows:

1. **Initialization**: Randomly select `k` points as initial centroids.

2. **Assignment Step**: Assign each data point to the nearest centroid, forming `k` clusters.

![Initial Clustering Step](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/kclusters_1.png)

3. **Update Step**: Update each centroid to the average position of all points assigned to it.

![Updating Centroids](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/kclusters_2.png)

4. **Iteration**: Repeat the assignment and update steps until convergence is reached.

### K-means for Non-separated Clusters

K-means isn't limited to datasets with well-defined clusters. It's also effective in situations where cluster boundaries are ambiguous:

- **Example**: Segmenting t-shirt sizes (Small, Medium, Large) even when there aren't clear separations in a dataset. The algorithm creates clusters based on the distribution of data points.

![T-shirt Size Clustering](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/t_shirt.png)

- **Market Segmentation**: This approach can be likened to market segmentation, where the goal is to tailor products to suit different subpopulations, even if clear divisions among these groups are not initially apparent.

### Optimization Objective

The goal of the K-means algorithm is to minimize the total sum of squared distances between data points and their respective cluster centroids. This objective leads to the formation of clusters that are as compact and distinct as possible given the chosen value of `k`.

### Tracking Variables in K-means

During the K-means process, two sets of variables are pivotal:

1. **Cluster Assignment ($c^i$)**: This denotes the cluster index (ranging from 1 to K) to which the data point $x^i$ is currently assigned.
2. **Centroid Location ($\mu_k$)**: This represents the location of centroid $k$.
3. **Centroid of Assigned Cluster ($\mu_{c^{(i)}}$)**: This is the centroid of the cluster to which the data point $x^i$ has been assigned.

### The Optimization Objective

The goal of K-means is formulated as minimizing the following cost function:

$$J(c^{(1)}, ..., c^{(m)}, \mu_1, ..., \mu_K) = \frac{1}{m} \sum_{i=1}^{m} ||x^{(i)} - \mu_{c^{(i)}}||^2$$

This function calculates the average of the squared distances from each data point to the centroid of its assigned cluster.

![Cost Function Visualization](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/cost_cluster.png)

### Algorithm Steps and Optimization

K-means consists of two steps, each minimizing the cost function in different respects:

1. **Cluster Assignment**: Minimizes $J(...)$ with respect to $c_1, c_2, ..., c_i$. This step finds the nearest centroid for each example, without altering the centroids.
2. **Move Centroid**: Minimizes $J(...)$ with respect to $\mu_k$. This step updates each centroid to the average position of all points assigned to it.

These steps are iterated until the algorithm converges, ensuring both parts of the algorithm work together to minimize the cost function.

### Random Initialization

K-means can converge to different solutions based on the initial placement of centroids.

![Different Convergence Outcomes](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/optimum_cluster.png)

- The algorithm is typically initialized randomly.
- It is run multiple times (say, 100 times) with different initializations.
- The clustering configuration with the lowest distortion (cost function value) at convergence is selected.

### The Elbow Method

Choosing the optimal number of clusters (K) is a challenge in K-means. The Elbow Method is a heuristic used to determine this:

- The cost function $J(...)$ is computed for a range of values of K.
- The cost typically decreases as K increases. The objective is to find a balance between the number of clusters and the minimization of the cost function.
- The "elbow" point in the plot of K vs $J(...)$ is considered a good choice for K. This is where the rate of decrease sharply changes.

![Elbow Method Graph](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/elbow.png)
