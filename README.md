# Machine_Learning Short notes (Revision)

1️⃣ Linear Regression 
->Used for predicting continuous values.
->Models the relationship between dependent and independent variables by fitting a linear equation.

2️⃣ Logistic Regression 
->Ideal for binary classification problems.
->Estimates the probability that an instance belongs to a particular class.

3️⃣ Decision Trees 
->Splits data into subsets based on the value of input features.
->Easy to visualize and interpret but can be prone to overfitting.

4️⃣ Random Forest 
->An ensemble method using multiple decision trees.
->Reduces overfitting and improves accuracy by averaging multiple trees.

5️⃣ Support Vector Machines (SVM) 
->Finds the hyperplane that best separates different classes.
->Effective in high-dimensional spaces and for classification tasks.

6️⃣ k-Nearest Neighbors (k-NN) 
->Classifies data based on the majority class among the k-nearest neighbors.
->Simple and intuitive but can be computationally intensive.

7️⃣ K-Means Clustering 
->Partitions data into k clusters based on feature similarity.
->Useful for market segmentation, image compression, and more.

8️⃣ Naive Bayes 
->Based on Bayes' theorem with an assumption of independence among predictors.
->Particularly useful for text classification and spam filtering.

9️⃣ Neural Networks 
->Mimic the human brain to identify patterns in data.
->Power deep learning applications, from image recognition to natural language processing.

🔟 Gradient Boosting Machines (GBM) 
->Combines weak learners to create a strong predictive model. 
->Used in various applications like ranking, classification, and regression.

![visualization_table](https://github.com/manav-888/Machine_Learning/assets/28830098/13dd46ef-a619-4a52-aafc-f7be6443c25e)

![2024-07-08 12 13 36](https://github.com/manav-888/Machine_Learning/assets/28830098/5430bfe3-e3e5-49b1-83c9-50e252b88fe1)


#### topics 
Day 1: Linear Regression
- Concept: Predict continuous values.
- Implementation: Ordinary Least Squares.
- Evaluation: R-squared, RMSE.

Day 2: Logistic Regression
- Concept: Binary classification.
- Implementation: Sigmoid function.
- Evaluation: Confusion matrix, ROC-AUC.

Day 3: Decision Trees
- Concept: Tree-based model for classification/regression.
- Implementation: Recursive splitting.
- Evaluation: Accuracy, Gini impurity.

Day 4: Random Forest
- Concept: Ensemble of decision trees.
- Implementation: Bagging.
- Evaluation: Out-of-bag error, feature importance.

Day 5: Gradient Boosting
- Concept: Sequential ensemble method.
- Implementation: Boosting.
- Evaluation: Learning rate, number of estimators.

Day 6: Support Vector Machines (SVM)
- Concept: Classification using hyperplanes.
- Implementation: Kernel trick.
- Evaluation: Margin maximization, support vectors.

Day 7: k-Nearest Neighbors (k-NN)
- Concept: Instance-based learning.
- Implementation: Distance metrics.
- Evaluation: k-value tuning, distance functions.

Day 8: Naive Bayes
- Concept: Probabilistic classifier.
- Implementation: Bayes' theorem.
- Evaluation: Prior probabilities, likelihood.

Day 9: k-Means Clustering
- Concept: Partitioning data into k clusters.
- Implementation: Centroid initialization.
- Evaluation: Inertia, silhouette score.

Day 10: Hierarchical Clustering
- Concept: Nested clusters.
- Implementation: Agglomerative method.
- Evaluation: Dendrograms, linkage methods.

Day 11: Principal Component Analysis (PCA)
- Concept: Dimensionality reduction.
- Implementation: Eigenvectors, eigenvalues.
- Evaluation: Explained variance.

Day 12: Association Rule Learning
- Concept: Discover relationships between variables.
- Implementation: Apriori algorithm.
- Evaluation: Support, confidence, lift.

Day 13: DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
- Concept: Density-based clustering.
- Implementation: Epsilon, min samples.
- Evaluation: Core points, noise points.

Day 14: Linear Discriminant Analysis (LDA)
- Concept: Linear combination for classification.
- Implementation: Fisher's criterion.
- Evaluation: Class separability.

Day 15: XGBoost
- Concept: Extreme Gradient Boosting.
- Implementation: Tree boosting.
- Evaluation: Regularization, parallel processing.

Day 16: LightGBM
- Concept: Gradient boosting framework.
- Implementation: Leaf-wise growth.
- Evaluation: Speed, accuracy.

Day 17: CatBoost
- Concept: Gradient boosting with categorical features.
- Implementation: Ordered boosting.
- Evaluation: Handling of categorical data.

Day 18: Neural Networks
- Concept: Layers of neurons for learning.
- Implementation: Backpropagation.
- Evaluation: Activation functions, epochs.

Day 19: Convolutional Neural Networks (CNNs)
- Concept: Image processing.
- Implementation: Convolutions, pooling.
- Evaluation: Feature maps, filters.

Day 20: Recurrent Neural Networks (RNNs)
- Concept: Sequential data processing.
- Implementation: Hidden states.
- Evaluation: Long-term dependencies.

Day 21: Long Short-Term Memory (LSTM)
- Concept: Improved RNN.
- Implementation: Memory cells.
- Evaluation: Forget gates, output gates.

Day 22: Gated Recurrent Units (GRU)
- Concept: Simplified LSTM.
- Implementation: Update gate.
- Evaluation: Performance, complexity.

Day 23: Autoencoders
- Concept: Data compression.
- Implementation: Encoder, decoder.
- Evaluation: Reconstruction error.

Day 24: Generative Adversarial Networks (GANs)
- Concept: Generative models.
- Implementation: Generator, discriminator.
- Evaluation: Adversarial loss.

Day 25: Transfer Learning
- Concept: Pre-trained models.
- Implementation: Fine-tuning.
- Evaluation: Domain adaptation.


Day 26: Reinforcement Learning
- Concept: Learning through interaction.
- Implementation: Q-learning.
- Evaluation: Reward function, policy.

Day 27: Bayesian Networks
- Concept: Probabilistic graphical models.
- Implementation: Conditional dependencies.
- Evaluation: Inference, learning.

Day 28: Hidden Markov Models (HMM)
- Concept: Time series analysis.
- Implementation: Transition probabilities.
- Evaluation: Viterbi algorithm.

Day 29: Feature Selection Techniques
- Concept: Improving model performance.
- Implementation: Filter, wrapper methods.
- Evaluation: Feature importance.

Day 30: Hyperparameter Optimization
- Concept: Model tuning.
- Implementation: Grid search, random search.
- Evaluation: Cross-validation.



# Linear Regression 
### Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

#### Example data
data = {
    'Size': [1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400],
    'Price': [300000, 320000, 340000, 360000, 380000, 400000, 420000, 440000, 460000, 480000]
}
df = pd.DataFrame(data)

#### Independent variable (feature) and dependent variable (target)
X = df[['Size']]
y = df['Price']

#### Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

##### Creating and training the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

##### Making predictions
y_pred = model.predict(X_test)

##### Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

##### Plotting the results
plt.scatter(X, y, color='blue')  # Original data points
plt.plot(X_test, y_pred, color='red', linewidth=2)  # Regression line
plt.xlabel('Size (sq ft)')
plt.ylabel('Price ($)')
plt.title('Linear Regression: House Prices vs Size')
plt.show()

### Explanation
1. Libraries: We import necessary libraries like numpy, pandas, sklearn, and matplotlib.
2. Data Preparation: We create a DataFrame containing the size and price of houses.
3. Feature and Target: We separate the feature (Size) and the target (Price).
4. Train-Test Split: We split the data into training and testing sets.
5. Model Training: We create a LinearRegression model and train it using the training data.
6. Predictions: We use the trained model to predict house prices for the test set.
7. Evaluation: We evaluate the model using Mean Squared Error (MSE) and R-squared (R²) metrics.
8. Visualization: We plot the original data points and the regression line to visualize the model's performance.

#### Evaluation Metrics

- Mean Squared Error (MSE): Measures the average squared difference between the actual and predicted values. Lower values indicate better performance.
- R-squared (R²): Represents the proportion of the variance in the dependent variable that is predictable from the independent variable(s). Values closer to 1 indicate a better fit.



## Logistic Regression
Logistic regression is used for binary classification problems, where the outcome is a categorical variable with two possible outcomes (e.g., 0 or 1, true or false). Instead of predicting a continuous value like linear regression, logistic regression predicts the probability of a specific class.

The logistic regression model uses the logistic function (also known as the sigmoid function) to map predicted values to probabilities. 

## Implementation

Let's consider an example using Python and its libraries.

## Example
Suppose we have a dataset that records whether a student has passed an exam based on the number of hours they studied.

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Example data
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Passed': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

# Independent variable (feature) and dependent variable (target)
X = df[['Hours_Studied']]
y = df['Passed']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Creating and training the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

# Evaluating the model
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)

print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")
print(f"ROC-AUC: {roc_auc}")

# Plotting the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()## Explanation of the Code

1. Libraries: We import necessary libraries like numpy, pandas, sklearn, and matplotlib.
2. Data Preparation: We create a DataFrame containing the hours studied and whether the student passed.
3. Feature and Target: We separate the feature (Hours_Studied) and the target (Passed).
4. Train-Test Split: We split the data into training and testing sets.
5. Model Training: We create a LogisticRegression model and train it using the training data.
6. Predictions: We use the trained model to predict the pass/fail outcome for the test set and also obtain the predicted probabilities.
7. Evaluation: We evaluate the model using the confusion matrix, classification report, and ROC-AUC score.
8. Visualization: We plot the ROC curve to visualize the model's performance.

## Evaluation Metric
- Confusion Matrix: Shows the counts of true positives, true negatives, false positives, and false negatives.
- Classification Report: Provides precision, recall, F1-score, and support for each class.
- ROC-AUC: Measures the model's ability to distinguish between the classes. AUC (Area Under the Curve) closer to 1 indicates better performance.
...

## Support Vector Machine

Concept: Support Vector Machines (SVM) are supervised learning models used for classification and regression tasks. The goal of SVM is to find the optimal hyperplane that maximally separates the classes in the feature space. The hyperplane is chosen to maximize the margin, which is the distance between the hyperplane and the nearest data points from each class, known as support vectors.

For nonlinear data, SVM uses a kernel trick to transform the input features into a higher-dimensional space where a linear separation is possible. Common kernels include:
- Linear Kernel
- Polynomial Kernel
- Radial Basis Function (RBF) Kernel
- Sigmoid Kernel

## Implementation Example
Suppose we have a dataset that records features like petal length and petal width to classify the species of iris flowers.

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Example data (Iris dataset)
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data[:, 2:4]  # Using petal length and petal width as features
y = iris.target

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Creating and training the SVM model with RBF kernel
model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=0)
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")

# Plotting the decision boundary
def plot_decision_boundary(X, y, model):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)

    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='bright', edgecolor='k', s=50)
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.title('SVM Decision Boundary')
    plt.show()

plot_decision_boundary(X_test, y_test, model)
#### Explanation of the Code

1. Importing Libraries
2. Data Preparation
3. Train-Test Split
4. Model Training: We create an SVC model with an RBF kernel (kernel='rbf'), regularization parameter C=1.0, and gamma parameter set to 'scale', and train it using the training data.
5. Predictions: We use the trained model to predict the species of iris flowers for the test set.
6. Evaluation: We evaluate the model using accuracy, confusion matrix, and classification report.
7. Visualization: Plot the decision boundary to visualize how the SVM separates the classes.

#### Decision Boundary

The decision boundary plot helps to visualize how the SVM model separates the different classes in the feature space. The SVM with an RBF kernel can capture more complex relationships than a linear classifier.

SVMs are powerful for high-dimensional spaces and effective when the number of dimensions is greater than the number of samples. However, they can be memory-intensive and require careful tuning of hyperparameters such as the regularization parameter \(C\) and kernel parameters.





