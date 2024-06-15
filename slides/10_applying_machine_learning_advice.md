## Advice for Applying Machine Learning Techniques and Debugging Learning Algorithms

### Troubleshooting High Error Rates

When facing high error rates with a machine learning model, especially when tested on new data, various strategies can be employed to diagnose and address the problem.

#### Steps for Improvement

1. **Add More Training Data:** Sometimes, the model has not seen enough examples to generalize well.
2. **Add or Remove Features:** The model might be missing key features that are important for predictions or might be overwhelmed by irrelevant or noisy features.
3. **Adjust Regularization Parameter ($\lambda$):** If the model is overfitting, increasing the regularization parameter can help. If underfitting, decreasing it might be beneficial.
4. **Data Splitting:** Divide your dataset into a training set and a test set to assess model performance on unseen data.
5. **Model Selection with Validation Set:** Create a training, validation, and test set to tune and select the best model.
6. **Polynomial Degree Analysis:** Plot errors for training and validation sets against polynomial degrees to diagnose high bias (underfitting) or high variance (overfitting).
7. **Advanced Optimization Algorithms:** Employ sophisticated optimization techniques to minimize the cost function more effectively.

### Debugging a Learning Algorithm

Consider a scenario where you have used regularized linear regression to predict home prices, but the model yields high errors on new data.

#### Cost Function

The cost function for the regularized linear regression:

$$J(\theta) = \frac{1}{2m} \left[ \sum_{i=1}^{m}(h_{\theta}(x^{(i)} - y^{(i)})^2 + \lambda \sum_{j=1}^{m} \theta_j^2 \right]$$

#### Next Steps for Improvement

1. **Obtain More Training Data:** Helps the algorithm to better generalize.
2. **Reduce Feature Set:** Try using fewer features to avoid overfitting.
3. **Incorporate More Features:** Add new features that might be relevant.
4. **Add Polynomial Features:** Enables the model to capture more complex relationships.
5. **Modify $\lambda$:** Adjust the regularization parameter to find the right balance between bias and variance.

### Evaluating the Hypothesis

Evaluating the model’s performance involves splitting the data and computing errors:

#### Data Split

- **Training Set:** Used to learn the parameters, $\theta$.
- **Test Set:** Used to compute the test error.

#### Test Error Calculation

Compute the test error using the learned parameters:

$$J_{test}(\theta) = \frac{1}{2m_{test}} \sum_{i=1}^{m_{test}}(h_{\theta}(x^{(i)}_{test} - y^{(i)}_{test})^2$$

This step helps to evaluate how well the model generalizes to new, unseen data.

### Model Selection Process

When applying machine learning algorithms, selecting the right model and parameters is crucial. This includes choosing the regularization parameter or the polynomial degree for regression models. The challenge is to balance between underfitting (high bias) and overfitting (high variance).

1. **Determine Polynomial Degree (`d`):** You may want to choose between different degrees of polynomial models, ranging from linear ($d=1$) to higher-degree polynomials.
   
For example, models can range from $h_{\theta}(x) = \theta_0 + \theta_1x$ to $h_{\theta}(x) = \theta_0 + ... + \theta_{10}x^{10}$.

2. **Train Models:** Train each model on the training dataset to obtain the parameter vector $\theta^d$ for each degree $d$.

3. **Compute Test Set Error:** Use $J_{test}(\theta^d)$ to evaluate the error on the test set for each model.

4. **Cross-Validation:** Test the models on a separate cross-validation set and compute the cross-validation error for each.

5. **Model Selection:** Select the model with the lowest cross-validation error.

### Error Metrics

I. Training Error:

$$J_{train}(\theta) = \frac{1}{2m}  \sum_{i=1}^{m}(h_{\theta}(x^{(i)} - y^{(i)})^2$$

II. Cross-Validation Error:

$$J_{cv}(\theta) = \frac{1}{2m_{cv}}  \sum_{i=1}^{m_{cv}}(h_{\theta}(x^{(i)}_{cv} - y^{(i)}_{cv})^2$$

III. Test Error:

$$J_{test}(\theta) = \frac{1}{2m_{test}}  \sum_{i=1}^{m_{test}}(h_{\theta}(x^{(i)}_{test} - y^{(i)}_{test})^2$$

### Diagnosing Bias vs. Variance

Diagnosing the nature of the error (high bias or high variance) can guide you in improving your model.

#### Visual Representation

![Bias vs Variance Diagnosis](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/diagnosis.png)

#### Plotting Error vs. Polynomial Degree

By plotting the training and cross-validation errors against the degree of the polynomial, you can visually assess the nature of the problem:

![Error vs Polynomial Degree](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/error_vs_d.png)

- **High Bias (Underfitting):** Both training and cross-validation errors are high. The model is too simple and does not capture the underlying trend in the data well.
  
- **High Variance (Overfitting):** Low training error but high cross-validation error. The model is too complex and captures noise in the training data, failing to generalize well.

### Regularized Linear Regression Model

Consider a high-order polynomial linear regression model:

$$h_{\theta}(x) = \theta_0 + \theta_1x + \theta_2x^2 + \theta_3x^3 + \theta_4x^4$$

The regularized cost function for this model is:

$$J(\theta) = \frac{1}{2m} \left[ \sum_{i=1}^{m}(h_{\theta}(x^{(i)} - y^{(i)})^2 + \lambda \sum_{j=1}^{n} \theta_j^2 \right]$$

- **$\lambda$:** Regularization parameter controlling the degree of regularization.
- **$\theta_j$:** Coefficients of the polynomial terms.
- **$m$:** Number of training examples.
- **$n$:** Number of features (polynomial terms in this case).

### Impact of the Regularization Parameter ($\lambda$)

The choice of $\lambda$ has a significant impact on the model's performance:

- **High $\lambda$ (High Bias):** Leads to underfitting as it overly penalizes the coefficients, resulting in a too simple model.
- **Intermediate $\lambda$:** Balances between bias and variance, often yielding a well-performing model.
- **Small $\lambda$ (High Variance):** Leads to overfitting as it does not sufficiently penalize the coefficients, allowing the model to fit too closely to the training data.

![Effect of Lambda on Regularization](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/lambda.png)

Try a range of $\lambda$ values, for each value minimize $J(\theta)$ to find $\theta^{(i)}$, and then compute the average squared error on the cross-validation set. Choose the $\lambda$ that results in the lowest error.

### Learning Curves

Learning curves plot the training error and cross-validation error against the number of training examples. They help diagnose bias and variance issues.

Plotting Learning Curves:

- **$J_{train}$ (Training Error):** Average squared error on the training set.
- **$J_{cv}$ (Cross-Validation Error):** Average squared error on the cross-validation set.
- **$m$:** Number of training examples.

![Learning Curve Visualization](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/learning_curve.png)

- **Small Training Sets:** $J_{train}$ tends to be lower because the model can fit small datasets well.
- **As $m$ Increases:** The model generalizes better, so $J_{cv}$ decreases.

#### Diagnosing Bias vs. Variance from Learning Curves

- **High Bias (Underfitting):** Both $J_{train}$ and $J_{cv}$ are high, and adding more training data does not significantly improve performance.
- **High Variance (Overfitting):** $J_{train}$ is low, but $J_{cv}$ is much higher. Increasing the training set size can help the model generalize better.

## Reference

These notes are based on the free video lectures offered by Stanford University, led by Professor Andrew Ng. These lectures are part of the renowned Machine Learning course available on Coursera. For more information and to access the full course, visit the [Coursera course page](https://www.coursera.org/learn/machine-learning).

