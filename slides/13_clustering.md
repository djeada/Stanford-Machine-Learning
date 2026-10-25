# Clustering in Unsupervised Learning

Clustering summarizes inputs by assigning observations to groups under a chosen notion of similarity. K-means represents each group by a centroid and minimizes squared Euclidean reconstruction distance. Its groups depend on feature scaling, the number of centroids, and initialization; they need a substantive interpretation after fitting.

## Overview of Clustering

I. **Objective**: To uncover hidden patterns and structures in unlabeled data.

II. **Applications**:

- Market segmentation, categorizing clients based on varying characteristics.
- Social network analysis to understand relationship patterns.
- Organizing computer clusters and data centers based on network structure and location.
- Analyzing astronomical data for insights into galaxy formation.

## The K-means Algorithm

K-means is a widely-used algorithm for clustering, celebrated for its simplicity and efficacy. It works as follows:

1. **Initialization**: Randomly select `k` points as initial centroids.

2. **Assignment Step**: Assign each data point to the nearest centroid, forming `k` clusters.

![Initial Clustering Step](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/kclusters_1.png)

3. **Update Step**: Update each centroid to the average position of all points assigned to it.

![Updating Centroids](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/kclusters_2.png)

4. **Iteration**: Repeat the assignment and update steps until convergence is reached.

## K-means for Non-separated Clusters

K-means isn't limited to datasets with well-defined clusters. It's also effective in situations where cluster boundaries are ambiguous:

- **Example**: Segmenting t-shirt sizes (Small, Medium, Large) even when there aren't clear separations in a dataset. The algorithm creates clusters based on the distribution of data points.

![T-shirt Size Clustering](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/t_shirt.png)

- **Market Segmentation**: This approach can be likened to market segmentation, where the goal is to tailor products to suit different subpopulations, even if clear divisions among these groups are not initially apparent.

## Optimization Objective

The goal of the K-means algorithm is to minimize the total sum of squared distances between data points and their respective cluster centroids. This objective leads to the formation of clusters that are as compact and distinct as possible given the chosen value of `k`.

## Tracking Variables in K-means

During the K-means process, two sets of variables are pivotal:

1. **Cluster Assignment ($c^i$)**: This denotes the cluster index (ranging from 1 to K) to which the data point $x^i$ is currently assigned.
2. **Centroid Location ($\mu_k$)**: This represents the location of centroid $k$.
3. **Centroid of Assigned Cluster ($\mu_{c^{(i)}}$)**: This is the centroid of the cluster to which the data point $x^i$ has been assigned.

## The Optimization Objective

The goal of K-means is formulated as minimizing the following cost function:

$$J(c^{(1)}, ..., c^{(m)}, \mu_1, ..., \mu_K) = \frac{1}{m} \sum_{i=1}^{m} ||x^{(i)} - \mu_{c^{(i)}}||^2$$

This function calculates the average of the squared distances from each data point to the centroid of its assigned cluster.

![Cost Function Visualization](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/cost_cluster.png)

## Algorithm Steps and Optimization

K-means consists of two steps, each minimizing the cost function in different respects:

1. **Cluster Assignment**: Minimizes $J(...)$ with respect to $c_1, c_2, ..., c_i$. This step finds the nearest centroid for each example, without altering the centroids.
2. **Move Centroid**: Minimizes $J(...)$ with respect to $\mu_k$. This step updates each centroid to the average position of all points assigned to it.

These steps are iterated until the algorithm converges, ensuring both parts of the algorithm work together to minimize the cost function.

## Random Initialization

K-means can converge to different solutions based on the initial placement of centroids.

![Different Convergence Outcomes](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/optimum_cluster.png)

- The algorithm is typically initialized randomly.
- It is run multiple times (say, 100 times) with different initializations.
- The clustering configuration with the lowest distortion (cost function value) at convergence is selected.

Here is a Python implementation that demonstrates the core steps of the K-means algorithm and illustrates the effect of random initialization on the final clustering outcome:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate mock data
np.random.seed(42)
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# K-means algorithm implementation
def initialize_centroids(X, k):
    return X[np.random.choice(X.shape[0], k, replace=False)]

def assign_clusters(X, centroids):
    return np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)

def move_centroids(X, labels, k, previous):
    return np.array([X[labels == i].mean(axis=0) if np.any(labels == i)
                     else previous[i] for i in range(k)])

def compute_cost(X, labels, centroids):
    return np.mean(np.sum((X - centroids[labels])**2, axis=1))

def k_means(X, k, max_iters=100):
    centroids = initialize_centroids(X, k)
    for _ in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = move_centroids(X, labels, k, centroids)
        converged = np.allclose(centroids, new_centroids, rtol=0, atol=1e-10)
        centroids = new_centroids
        if converged:
            break
    labels = assign_clusters(X, centroids)
    cost = compute_cost(X, labels, centroids)
    return labels, centroids, cost

# Running K-means multiple times with random initializations
def run_k_means(X, k, num_runs=100):
    best_labels = None
    best_centroids = None
    lowest_cost = float('inf')

    for _ in range(num_runs):
        labels, centroids, cost = k_means(X, k)
        if cost < lowest_cost:
            lowest_cost = cost
            best_labels = labels
            best_centroids = centroids

    return best_labels, best_centroids, lowest_cost

# Parameters
k = 4
num_runs = 100

# Run K-means
best_labels, best_centroids, lowest_cost = run_k_means(X, k, num_runs)

# Plotting the result
plt.scatter(X[:, 0], X[:, 1], c=best_labels, s=50, cmap='viridis')
plt.scatter(best_centroids[:, 0], best_centroids[:, 1], s=200, c='red', marker='X')
plt.title(f'Best K-means clustering with k={k} (Lowest Cost: {lowest_cost:.2f})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
```

Below is a static plot showing the best clustering found with $k=4$. Different colors represent the clusters, and the red 'X' markers indicate the centroids. This configuration has the lowest cost after running the algorithm 100 times with various random initializations.

![Figure_1](https://github.com/djeada/Stanford-Machine-Learning/assets/37275728/b10c5238-7449-4a3c-a79f-53036f0adc09)

## The Elbow Method

Choosing the optimal number of clusters (K) is a challenge in K-means. The Elbow Method is a heuristic used to determine this:

- The cost function $J(...)$ is computed for a range of values of K.
- The cost typically decreases as K increases. The objective is to find a balance between the number of clusters and the minimization of the cost function.
- The "elbow" point in the plot of K vs $J(...)$ is considered a good choice for K. This is where the rate of decrease sharply changes.

![Elbow Method Graph](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/elbow.png)

Here is a Python implementation for calculating and plotting the cost for a range of $k$ values to identify the optimal number of clusters:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate mock data
np.random.seed(42)
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# K-means algorithm implementation
def initialize_centroids(X, k):
    return X[np.random.choice(X.shape[0], k, replace=False)]

def assign_clusters(X, centroids):
    return np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)

def move_centroids(X, labels, k, previous):
    return np.array([X[labels == i].mean(axis=0) if np.any(labels == i)
                     else previous[i] for i in range(k)])

def compute_cost(X, labels, centroids):
    return np.mean(np.sum((X - centroids[labels])**2, axis=1))

def k_means(X, k, max_iters=100):
    centroids = initialize_centroids(X, k)
    for _ in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = move_centroids(X, labels, k, centroids)
        converged = np.allclose(centroids, new_centroids, rtol=0, atol=1e-10)
        centroids = new_centroids
        if converged:
            break
    labels = assign_clusters(X, centroids)
    cost = compute_cost(X, labels, centroids)
    return labels, centroids, cost

# Running K-means multiple times with random initializations
def run_k_means(X, k, num_runs=100):
    best_labels = None
    best_centroids = None
    lowest_cost = float('inf')

    for _ in range(num_runs):
        labels, centroids, cost = k_means(X, k)
        if cost < lowest_cost:
            lowest_cost = cost
            best_labels = labels
            best_centroids = centroids

    return best_labels, best_centroids, lowest_cost

# Define the function to compute the cost for different values of k
def compute_cost_for_k(X, k_values, num_runs=10):
    costs = []
    for k in k_values:
        _, _, lowest_cost = run_k_means(X, k, num_runs)
        costs.append(lowest_cost)
    return costs

# Define the range of k values to test
k_values = range(1, 11)

# Compute the costs for each k
costs = compute_cost_for_k(X, k_values)

# Plot the Elbow Method graph
plt.plot(k_values, costs, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Cost function (J)')
plt.title('Elbow Method for Determining Optimal k')
plt.xticks(k_values)
plt.show()
```

Below is a static plot illustrating the Elbow Method for determining the optimal number of clusters ($k$). The plot shows the cost function $J$ as a function of $k$. The "elbow" point, where the rate of decrease in cost significantly changes, is a good indicator of the optimal $k$.

![Figure_1](https://github.com/djeada/Stanford-Machine-Learning/assets/37275728/ec42478e-bbfa-492f-9ade-8a760455b3de)

## Derive Both Steps from the Same Objective

With centroids fixed, each assignment minimizes its own squared-distance term by choosing the nearest centroid. With assignments fixed, a nonempty cluster's objective is $\sum_{i:c_i=k}\|x_i-\mu_k\|^2$. Differentiating gives

$$
2\sum_{i:c_i=k}(\mu_k-x_i)=0,
\qquad
\mu_k=\frac{1}{m_k}\sum_{i:c_i=k}x_i.
$$

The mean is therefore the exact minimizer for that step. Replacing squared Euclidean distance by absolute distance would instead lead to coordinate medians. A different distance and unchanged mean update do not automatically optimize one coherent objective.

Each alternating step cannot increase the objective, but the joint problem is nonconvex. Convergence to a stable assignment does not prove the best possible partition. Multiple starts improve the search, and a deterministic tie rule makes ambiguous assignments reproducible.

## Worked Assignment and Centroid Update

Take one-dimensional observations $0,2,8,10$ with initial centers zero and ten. Their squared distances to the two centers are $(0,100),(4,64),(64,4),(100,0)$. The first two join the first cluster and the other two the second.

Updated means are $(0+2)/2=1$ and $(8+10)/2=9$. The squared-distance sum drops from eight to four. The course's average distortion is then $4/4=1$, while scikit-learn reports the sum as inertia.

```python
import numpy as np
from sklearn.cluster import KMeans
X = np.array([[0.0], [2.0], [8.0], [10.0]])
initial = np.array([[0.0], [10.0]])

def assign(X, centers):
    distances = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return distances.argmin(axis=1)

def update(X, labels, previous):
    centers = previous.copy()
    for k in range(len(centers)):
        members = X[labels == k]
        if len(members):
            centers[k] = members.mean(axis=0)
    return centers

def inertia(X, centers):
    labels = assign(X, centers)
    return np.sum((X - centers[labels]) ** 2)
labels = assign(X, initial)
centers = update(X, labels, initial)
np.testing.assert_array_equal(labels, [0, 0, 1, 1])
np.testing.assert_allclose(centers, [[1.0], [9.0]])
np.testing.assert_allclose([inertia(X, initial), inertia(X, centers)], [8.0, 4.0])
library = KMeans(n_clusters=2, init=initial, n_init=1, algorithm='lloyd', tol=1e-12).fit(X)
np.testing.assert_allclose(np.sort(library.cluster_centers_[:, 0]), [1.0, 9.0])
np.testing.assert_allclose(library.inertia_, inertia(X, centers))
empty = np.array([[0.0], [10.0], [100.0]])
assert np.isfinite(update(X, assign(X, empty), empty)).all()
print('assignments:', labels, 'updated centers:', centers.ravel())
print('inertia/course distortion:', library.inertia_, library.inertia_ / len(X))
```

The comparison uses identical initial centers and one initialization, isolating the algorithm from randomness. Sorting the centroids permits comparison without relying on arbitrary cluster IDs. In general, swapping all cluster labels preserves the same partition and objective.

An empty cluster has no mean. This example deliberately retains its previous center, making the behavior explicit and finite. Another implementation can relocate it, so empty-cluster cases need not match this policy. Final labels are recomputed from final centers, including when an iteration limit is reached.

## Interpret Distortion and Choose K for the Task

The globally minimal squared distortion cannot increase when K increases, since an extra center can reproduce or improve an existing solution. Separate local optimization runs can violate an apparent monotone curve because they find different local solutions. More restarts or comparable initialization helps distinguish a search problem from a change in the objective.

An elbow is a visual tradeoff, not an identified true number of populations. For product sizes, compare manufacturing constraints and coverage; for compression, compare reconstruction error and storage. A cluster pattern can be stable under the objective and still be unsuitable for the intended decision.

Squaring distances emphasizes outliers. Features measured on large numerical scales dominate raw Euclidean distance. Standardization changes the geometry and should reflect the question, while fitting any learned transform only on permitted training data. Spherical compact groups are a preference of the squared-distance objective; curved, unequal-density, or strongly elongated groups may need a different model.

For a two-dimensional point with feature changes $(1,100)$, raw squared distance is $10{,}001$. If the training feature scales are $(1,100)$, the standardized changes are $(1,1)$ with squared distance two. The same records can therefore produce a different partition without any error in either implementation.

## Reference

These notes are based on the free video lectures offered by Stanford University, led by Professor Andrew Ng. These lectures are part of the renowned Machine Learning course available on Coursera. For more information and to access the full course, visit the [Coursera course page](https://www.coursera.org/learn/machine-learning).
