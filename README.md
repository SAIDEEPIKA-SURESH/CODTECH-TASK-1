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

Output

 SquareFootage  NumBedrooms  Location
0           1360            2  suburban
1           4272            3     urban
2           3592            1     rural
3            966            6     rural
4           4926            1     rural

SquareFootage  NumBedrooms  Location_suburban  Location_urban  \
0           1360            2               True           False   
1           4272            3              False            True   
2           3592            1              False           False   
3            966            6              False           False   
4           4926            1              False           False   

      HousePrice  
0  314267.220307  
1  861428.341362  
2  617443.402870  
3  439469.510490  
4  816931.314750  
SquareFootage          int64
NumBedrooms            int64
Location_suburban       bool
Location_urban          bool
HousePrice           float64
dtype: object

const                float64
SquareFootage          int64
NumBedrooms            int64
Location_suburban       bool
Location_urban          bool
dtype: object
     const  SquareFootage  NumBedrooms  Location_suburban  Location_urban
29     1.0           2028            1              False           False
535    1.0           3519            4               True           False
695    1.0           4507            4               True           False
557    1.0           3371            1              False            True
836    1.0           2871            5               True           False
29     369708.909415
535    754229.877227
695    879772.770840
557    672499.580891
836    687666.902315
Name: HousePrice, dtype: float64

[12]
0s

const                float64
SquareFootage        float64
NumBedrooms          float64
Location_suburban    float64
Location_urban       float64
dtype: object
float64
const                True
SquareFootage        True
NumBedrooms          True
Location_suburban    True
Location_urban       True
dtype: bool
True

True
True
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.990
Model:                            OLS   Adj. R-squared:                  0.990
Method:                 Least Squares   F-statistic:                 1.951e+04
Date:                Wed, 24 Jul 2024   Prob (F-statistic):               0.00
Time:                        00:06:23   Log-Likelihood:                -9088.8
No. Observations:                 800   AIC:                         1.819e+04
Df Residuals:                     795   BIC:                         1.821e+04
Df Model:                           4                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       2524.0216   2626.319      0.961      0.337   -2631.318    7679.362
x1           149.6494      0.589    253.991      0.000     148.493     150.806
x2          4.999e+04    431.983    115.712      0.000    4.91e+04    5.08e+04
x3          3.039e+04   1813.144     16.759      0.000    2.68e+04    3.39e+04
x4          8.069e+04   1821.866     44.290      0.000    7.71e+04    8.43e+04
==============================================================================
Omnibus:                        0.733   Durbin-Watson:                   2.013
Prob(Omnibus):                  0.693   Jarque-Bera (JB):                0.648
Skew:                          -0.067   Prob(JB):                        0.723
Kurtosis:                       3.042   Cond. No.                     1.23e+04
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 1.23e+04. This might indicate that there are
strong multicollinearity or other numerical problems.

Testing MSE: 457245373.1531
Testing R-squared: 0.9900

![Screenshot 2024-07-24 055028](https://github.com/user-attachments/assets/a72ff791-0d85-4995-9f1b-0b87aba978dd)



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
