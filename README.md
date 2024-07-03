# Machine_Learning Short notes 




### Linear Regression 
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Example data
data = {
    'Size': [1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400],
    'Price': [300000, 320000, 340000, 360000, 380000, 400000, 420000, 440000, 460000, 480000]
}
df = pd.DataFrame(data)

# Independent variable (feature) and dependent variable (target)
X = df[['Size']]
y = df['Price']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Creating and training the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Plotting the results
plt.scatter(X, y, color='blue')  # Original data points
plt.plot(X_test, y_pred, color='red', linewidth=2)  # Regression line
plt.xlabel('Size (sq ft)')
plt.ylabel('Price ($)')
plt.title('Linear Regression: House Prices vs Size')
plt.show()


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
