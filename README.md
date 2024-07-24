Personal Information
Name: Sai Deepika S
Background: Student in an engineering college
Interests: Learning new technologies, fast-paced learner
Domain : Machine Learning
ID : CT6ML501





Project: Linear Regression on Housing Prices
Objective
To implement a linear regression model to predict housing prices based on features such as square footage, number of bedrooms, and location.

Steps Taken
Dataset Creation/Loading

Generated a synthetic dataset with features: square footage, number of bedrooms, and location.
Data Preprocessing

Categorical Conversion: Converted categorical variables (e.g., location) to numeric using one-hot encoding.
Feature and Target Separation: Split the data into features (X) and target variable (y).
Data Splitting: Divided the dataset into training and testing sets using train_test_split.
Model Training

Added a constant term to the feature set for the intercept.
Trained a linear regression model using statsmodels' OLS method.
Model Evaluation

Predicted house prices on the test set.
Calculated performance metrics including Mean Squared Error (MSE) and R-squared.
Visualization

Created scatter plots to compare actual vs. predicted values.
Added a diagonal line for visual reference of perfect predictions.
Code Summary
python
Copy code
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Data creation
np.random.seed(42)
df = pd.DataFrame({
    'SquareFootage': np.random.randint(500, 5000, 1000),
    'NumBedrooms': np.random.randint(1, 7, 1000),
    'Location': np.random.choice(['urban', 'suburban', 'rural'], 1000)
})
df = pd.get_dummies(df, columns=['Location'], drop_first=True)
df['HousePrice'] = (150 * df['SquareFootage'] +
                    50000 * df['NumBedrooms'] +
                    80000 * df['Location_urban'] +
                    30000 * df['Location_suburban'] +
                    np.random.normal(scale=20000, size=1000))

# Preprocessing
X = df.drop('HousePrice', axis=1)
y = df['HousePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

# Model training
model = sm.OLS(y_train, X_train).fit()
print(model.summary())

# Evaluation
y_test_pred = model.predict(X_test)
mse_test = np.mean((y_test - y_test_pred) ** 2)
r2_test = 1 - (np.sum((y_test - y_test_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
print(f'Testing MSE: {mse_test:.4f}')
print(f'Testing R-squared: {r2_test:.4f}')

# Visualization
plt.scatter(y_test, y_test_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')
plt.show()
Use Cases
Real Estate Market Analysis: Helps predict property values for buying, selling, and investing.
Urban Planning: Assists in understanding property values in different locations for development projects.
Insurance: Evaluates risk and pricing for property insurance.
