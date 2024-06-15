## Machine Learning System Design: Building a Spam Classifier

These notes outline the key strategies and considerations for developing a spam classification system. This process involves several steps, from feature selection to error analysis, and addresses the challenges of working with skewed datasets.

### Prioritizing Work in Spam Classification

When building a spam classifier, a critical step is to select and utilize features effectively. Let's break down the process and include an example of how inputs might look.

![Spam Classifier Visualization](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/spam.png)

#### Feature Selection

The first step involves choosing words that are strongly indicative of whether an email is spam. This is based on the assumption that certain words appear more frequently in spam emails than in non-spam emails.

- **Spam Indicators:** Words like "buy", "discount", and "deal" are often used in promotional or spam emails.
- **Non-Spam Indicators:** Words like "Andrew" (a common name) or "now" might be more frequent in regular, non-spam emails.

#### Vector Representation of Emails

Each email is represented as a vector, where each element corresponds to one of the selected features (words).

- **Binary Encoding:** For each word in our feature set, if the word is present in the email, that feature is marked with a 1; if it's absent, it's marked with a 0.
  
#### Example of Input Representation

Consider an example where our feature set consists of words: `["buy", "discount", "deal", "Andrew", "now"]`. An email's content will be transformed into a vector based on the presence or absence of these words.

| Email Content                | buy | discount | deal | Andrew | now |
|------------------------------|-----|----------|------|--------|-----|
| "Buy now at a great discount"| 1   | 1        | 0    | 0      | 1   |
| "Meeting Andrew now"         | 0   | 0        | 0    | 1      | 1   |
| "Deal of the century"        | 0   | 0        | 1    | 0      | 0   |

#### Word Occurrences Analysis

The next step involves analyzing the frequency of each category of words (spam or non-spam) in your training dataset. This analysis can reveal:

- **Commonalities in Spam:** If certain words are predominantly present in spam emails.
- **Differentiators for Non-Spam:** Words that are usually absent in spam but present in non-spam emails.

#### Model Selection

Based on this vector representation, a classification model like logistic regression or a naive Bayes classifier can be employed. The choice of model might depend on:

- **Dataset Size:** Larger datasets might benefit from more complex models.
- **Feature Interpretability:** If understanding why an email is classified as spam is important, a model that offers interpretability (like decision trees) might be preferable.
- **Performance Metrics:** Based on the desired balance between precision and recall (as spam classification often deals with imbalanced datasets).

#### Why This Approach?

This method of feature selection and vector representation is effective for several reasons:

- **Simplicity:** It’s straightforward to implement and understand.
- **Scalability:** Can easily handle large volumes of data typical for email.
- **Adaptability:** The model can be updated as new spam trends emerge by adjusting the feature set.

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

