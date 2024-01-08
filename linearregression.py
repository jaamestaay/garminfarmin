import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# Read and format Independent Variables
calories_df = pd.read_csv("datasets/calories.csv")
steps_df = pd.read_csv("datasets/steps.csv")
stress_df = pd.read_csv("datasets/stress.csv")
X_df = pd.concat([calories_df['Active Calories'],
                  steps_df['Actual'],
                  stress_df['Stress']], axis=1)
X_df = X_df.rename(columns={'Actual': 'Steps'})
X = X_df.values
# Read and format Dependent Variable
rhr_df = pd.read_csv("datasets/rhr.csv")
y = rhr_df["Resting Heart Rate"].values

seed = 42
# Perform Regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
for value in range(len(y_pred)):
    print(y_pred[value], y_test[value])
print(reg.score(X_test, y_test))
print(mean_squared_error(y_test, y_pred, squared=False))

# Lasso Feature Selection
names = X_df.columns
lasso = Lasso(alpha=0.1)
lasso_coef = lasso.fit(X, y).coef_
plt.bar(names, lasso_coef)
plt.xticks(rotation=45)
plt.show()

predictions = reg.predict(X_test)