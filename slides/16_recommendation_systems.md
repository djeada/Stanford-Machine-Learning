## Recommendation Systems

Recommendation systems are a fundamental component in the interface between users and large-scale content providers like Amazon, eBay, and iTunes. These systems personalize user experiences by suggesting products, movies, or content based on past interactions and preferences.

### Concept of Recommender Systems

- **Purpose**: To recommend new content or products to users based on their historical interactions or preferences.
- **Nature**: Not a single algorithm, but a category of methods tailored to predicting user preferences.

### Example Scenario: Movie Rating Prediction

Imagine a company that sells movies and allows users to rate them. Consider the following setup:

- **Movies**: A catalog of five movies.
- **Users**: A database of four users.
- **User Ratings**: Users rate movies on a scale of 1 to 5.

![Movie Recommendation Example](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/movie_recommendation.png)

Notations:

- $n_u$: Number of users.
- $n_m$: Number of movies.
- $r(i, j)$: Indicator (1 or 0) if user $j$ has rated movie $i$.
- $y^{(i, j)}$: Rating given by user $j$ to movie $i$.

### Feature Vectors and Parameter Vectors

- **Movie Features**: Each movie can have a feature vector representing various attributes or genres.
- **Additional Feature (x0 = 1)**: For computational convenience, an additional feature is added to each movie's feature vector.

Example for "Cute Puppies of Love":
  
$$
x^{(3)} = \begin{bmatrix}
1 \\
0.99 \\
0
\end{bmatrix}
$$

- **User Parameters**: Each user has a parameter vector representing their preferences.

Example for user 1 (Alice) and her preference for "Cute Puppies of Love":

$$
\theta^{(1)} = \begin{bmatrix}
0 \\
5 \\
0
\end{bmatrix}
$$

### Making Predictions

To predict how much Alice might like "Cute Puppies of Love", we compute:

$$
(\theta^{(1)})^T x^{(3)} = (0 \cdot 1) + (5 \cdot 0.99) + (0 \cdot 0) = 4.95
$$

This predicts a high rating of 4.95 out of 5 for Alice for this movie, based on her preference parameters and the movie's features.

### Collaborative Filtering

- **Method**: Collaborative filtering is often used in recommender systems. It involves learning either user preferences or item features, depending on the data available.
- **Learning**: This can be achieved using algorithms like gradient descent.
- **Advantage**: The system can learn to make recommendations on its own without explicit programming of the features or preferences.

## Learning User Preferences in Recommendation Systems

In recommendation systems, learning user preferences $(\theta^j)$ is a key component. This process is akin to linear regression, but with a unique approach tailored for recommendation contexts.

### Minimizing Cost Function for User Preferences

- The objective is to minimize the following cost function for each user $j$:

$$
\min_{\theta^j} = \frac{1}{2m^{(j)}} \sum_{i:r(i,j)=1} \left((\theta^{(j)})^T(x^{(i)}) - y^{(i,j)}\right)^2 + \frac{\lambda}{2m^{(j)}} \sum_{k=1}^{n} (\theta_k^{(j)})^2
$$
  
- Here, $m^{(j)}$ is the number of movies rated by user $j$, $\lambda$ is the regularization parameter, and $r(i, j)$ indicates whether user $j$ has rated movie $i$.

### Gradient Descent for Optimization

Update Rule for $\theta_k^{(j)}$:

- For $k = 0$ (bias term):

$$
\theta_k^{(j)} := \theta_k^{(j)} - \alpha \sum_{i:r(i,j)=1} \left((\theta^{(j)})^T(x^{(i)}) - y^{(i,j)}\right) x_k^{(i)}
$$

- For $k \neq 0$:

$$
\theta_k^{(j)} := \theta_k^{(j)} - \alpha \left(\sum_{i:r(i,j)=1} \left((\theta^{(j)})^T(x^{(i)}) - y^{(i,j)}\right) x_k^{(i)} + \lambda \theta_k^{(j)}\right)
$$

### Collaborative Filtering Algorithm

Collaborative filtering leverages the interplay between user preferences and item (movie) features:

- **Learning Features from Preferences**: Given user preferences $(\theta^{(1)}, ..., \theta^{(n_u)})$, the algorithm can learn movie features $(x^{(1)}, ..., x^{(n_m)})$.
- **Cost Function for Movies**:

$$
\min_{x^{(1)}, ..., x^{(n_m)}} \frac{1}{2} \sum_{i=1}^{n_m} \sum_{i:r(i,j)=1} \left((\theta^{(j)})^T(x^{(i)}) - y^{(i,j)}\right)^2 + \frac{\lambda}{2} \sum_{i=1}^{n_m} \sum_{k=1}^{n} (x_k^{(i)})^2
$$

- **Iterative Process**: The algorithm typically involves an iterative process, alternating between optimizing for movie features and user preferences.

### Vectorization: Low Rank Matrix Factorization

- **Matrix Y**: Organize all user ratings into a matrix Y, which represents the interactions between users and movies.

Example $[5 \times 4]$ Matrix Y for 5 movies and 4 users:
  
$$
Y = \begin{pmatrix}
5 & 5 & 0 & 0 \\
5 & ? & ? & 0 \\
? & 4 & 0 & ? \\
0 & 0 & 5 & 4 \\
0 & 0 & 5 & 0
\end{pmatrix}
$$

- **Predicted Ratings Matrix**: The predicted ratings can be expressed as the product of matrices $X$ and $\Theta^T$.

$$
X \Theta^T = \begin{pmatrix}
(\theta^{(1)})^T(x^{(1)})   & \dots & (\theta^{(n_u)})^T(x^{(1)})   \\
\vdots                      & \ddots & \vdots                        \\
(\theta^{(1)})^T(x^{(n_m)}) & \dots & (\theta^{(n_u)})^T(x^{(n_m)})
\end{pmatrix}
$$

- **Matrix X**: Contains the features for each movie, stacked in rows.
- **Matrix $\Theta$**: Contains the user preferences, also stacked in rows.

### Scenario: A User with No Ratings

Imagine a user in a movie recommendation system who hasn't rated any movies. This scenario presents a challenge for typical collaborative filtering algorithms.

![User with No Ratings](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/no_ratings.png)

- For such a user, there are no movies for which $r(i, j) = 1$.
- As a result, the algorithm ends up minimizing only the regularization term for this user, which doesn't provide meaningful insight for recommendations.

### Mean Normalization Approach

1. **Compute the Mean Rating**: Calculate the average rating for each movie and store these in a vector $\mu$.
   
Example $\mu$ vector for a system with 5 movies:

$$
\mu = \begin{bmatrix}
 2.5  \\
 2.5  \\
 2    \\
 2.25 \\
 1.25
\end{bmatrix}
$$

2. **Normalize the Ratings Matrix**: Subtract the mean rating for each movie from all its ratings in the matrix $Y$.

Normalized Ratings Matrix $Y$:

$$
Y = \begin{pmatrix}
 2.5   & 2.5   & -2.5 & -2.5  & ? \\
 2.5   & ?     & ?    & -2.5  & ? \\
 ?     & 2     & -2   & ?     & ? \\
 -2.25 & -2.25 & 2.75 & 1.75  & ? \\
 -1.25 & -1.25 & 3.75 & -1.25 & ?
\end{pmatrix}
$$

3. **Adjust for Users with No Ratings**: For a new user with no ratings, their predicted rating for each movie can be initialized to the mean rating of that movie. This provides a baseline from which personalized recommendations can evolve as the user starts rating movies.

### Benefits of Mean Normalization

- **Handling New Users**: Provides a starting point for recommendations for users who have not yet provided any ratings.
- **Rating Sparsity**: Mitigates issues arising from sparse data, which is common in many real-world recommendation systems.
- **Balancing the Dataset**: Normalization helps in balancing the dataset, especially when there are variations in the number of ratings per movie.
