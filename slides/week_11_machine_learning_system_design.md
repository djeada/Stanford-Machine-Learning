## Machine Learning System Design: Building a Spam Classifier

These notes outline the key strategies and considerations for developing a spam classification system. This process involves several steps, from feature selection to error analysis, and addresses the challenges of working with skewed datasets.

### Prioritizing Work in Spam Classification

1. **Feature Selection:** Select a set of words that are indicative of spam or non-spam emails. For instance:
   - Words like "buy", "discount", and "deal" might suggest spam.
   - Words like "Andrew" and "now" might indicate non-spam.
2. **Vector Representation:** Each chosen word forms a feature in a long vector representing an email. A value of 1 is assigned if the word appears in the email, and 0 otherwise.
3. **Word Occurrences:** Analyze which category of words (spam or non-spam) occurs more frequently.

![Spam Classifier Visualization](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/spam.png)

### Improving System Accuracy

- **Collect More Data:** More data can help the system better learn the distinction between spam and non-spam.
- **Develop Sophisticated Features:** Include features based on email routing data or develop algorithms to detect common misspellings in spam.
- **Learning Curves:** Plot learning curves to assess whether additional data or features will benefit the system.

### Error Analysis

- **Manual Review of Errors:** Examine instances where the algorithm erred, particularly on the cross-validation set, to understand the nature of these errors.
- **Feature Analysis:** Determine what additional features could have helped correctly classify the challenging cases.

### Error Metrics for Skewed Classes

In cases where one class (e.g., non-spam) significantly outnumbers the other (e.g., spam), standard error metrics can be misleading.

#### Cancer Classification Example

- **Error Rate Deception:** An error rate of 1% might seem low, but if only 0.5% of the samples are cancer-positive, this error rate is not as impressive.
- **Importance of Precision and Recall:** In skewed datasets, precision and recall become critical metrics for assessing performance.

#### F-Score Computation

The F-Score is a way to combine precision and recall into a single metric, often used to choose the threshold that maximizes this score in cross-validation.

### Understanding Precision and Recall

Precision and recall are two critical metrics in classification tasks, especially when dealing with imbalanced datasets.

#### Definition of Terms

- **True Positive (TP):** Correctly identified positive.
- **False Positive (FP):** Incorrectly identified positive.
- **True Negative (TN):** Correctly identified negative.
- **False Negative (FN):** Incorrectly identified negative.

#### Precision

Precision measures the accuracy of the positive predictions. It is defined as:

$$\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$$

#### Recall

Recall measures how well the model identifies actual positives. It is defined as:

$$\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$$

### Trading off Precision and Recall

In many applications, there is a trade-off between precision and recall. Adjusting the threshold in a classifier can shift the balance between these two metrics.

#### Logistic Regression Classifier Example

- **Standard Threshold ($h_{\theta}(x) \geq 0.5$):** The default threshold for binary classification.
- **Modified Threshold ($h_{\theta}(x) \geq 0.8$):** Increases precision but may reduce recall, leading to fewer overall positive predictions.

![Precision-Recall Trade-off](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/precission_recall.png)

### Calculating the F-Score

The $F_{score}$ is a metric that combines precision and recall into a single number, often used to find a balance between these two measures:

$$F_{score} = 2 \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

- **Balanced Measure:** It harmonizes the precision and recall, especially useful when one is significantly lower than the other.
- **Threshold Selection:** One approach to finding the optimal threshold is to test various values and select the one that maximizes the $F_{score}$ on a cross-validation set.

