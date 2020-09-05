from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

# Single linear regression
lm = LinearRegression()
lm2 = LinearRegression()
path='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
df = pd.read_csv(path)

x = df[['highway-mpg']]
y = df[['price']]

lm.fit(x, y)
output = lm.predict(x)
print("Intercept = "+str(lm.intercept_)+" Slope = "+str(lm.coef_))

# Multiple Lin regression
z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
lm2.fit(z, y)
output2 = lm2.predict(z)
print("Intercept = "+str(lm2.intercept_)+" Slope = "+str(lm2.coef_))

# Shows difference between predicted value and actual value for the X value
# In this case shows how price differs for each value of highway mpg which is the X value
ax1 = sns.residplot(x=df['highway-mpg'],y=df['price'])
ax1.set(ylabel='Residual', title = "Highway-Price Residual Plot")
plt.show()
# Distribution plots show the difference between the predicted model and the actual values
# The closer the two graphs are to each other the more accurate the model is
ax = sns.distplot(a=df['price'], label= "Actual Price", color='r', hist=False)
# Distribution plot based on 1 variable
#sns.distplot(a=output, label="Predicted Price", color='b', hist=False,ax=ax).set_title("Highway-Price Dist Plot")
#plt.show()
# Distribution plot based on 4 variables
#sns.distplot(a=output2, label="Predicted Price", color='b', hist=False,ax=ax).set_title("Multiple-Price Dist Plot")
#plt.show()

# Polynomial regression in numpy - only 2 variables
f = np.polyfit(df['highway-mpg'], df['price'], 3)
p = np.poly1d(f)
print(p)
print(p(20))

# Pipeline - Makes polynomial process easier - Multivariable
Input = [('scale', StandardScaler()), ('polynomial', PolynomialFeatures(degree=3)), ('mode', LinearRegression())]
pipe = Pipeline(Input)
pipe.fit(z, y)

prediction = pipe.predict(X=z)

sns.distplot(a=prediction, label="Predicted Price", color='b', hist=False,ax=ax).set_title("Multiple-Price Dist Plot")
plt.show()

# Can visualize accuracy of model but to have a metric can use MSE (Mean squared error)
# This is the variance of actual and predicted squared / by num of points
print(mean_squared_error(df['price'], output2))

# Coefficient of Determination = R^2
# How close data is to fitted regression line
# The closer it is to one the better
print(lm.score(df[['highway-mpg']], df['price']))

# Prediction and Decision Making
# Creates a numpy array from the start to the end-1
test_input = np.array(1,101,1).reshape(-1,1)

# First step is using a regression plot to determine corr
# Next look at residual plot - Observe for nonlinear behaivor
# If not than Linear reg is good?
# A dist plot is good for multiple linear reg