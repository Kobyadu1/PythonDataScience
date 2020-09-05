import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# Function to plot polynomial models
def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()


def MPlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 100, 86)
    x_new_list = list()
    for x in x_new:
        x_new_list.append(int(x))

    y_new = model[x_new_list]

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()


path = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
df = pd.read_csv(path)

# Simple Linear Regression - Relantionship between
# Predictor/independent variable - X
# Pedicted/dependent variable - Y
# in format Y = mx + b

lm = LinearRegression()

# To see how highway mpg can help to predict the price
X = df[['highway-mpg']]
Y = df['price']

# First fit the model
lm.fit(X, Y)
lm.predict(X)

print("Highway-mpg")
# Can return intercept and slope with following command
print("Simple Linear Intercept "+str(lm.intercept_))
print("Simple Linear slope "+str(lm.coef_))

# Using engine size
print("\nEngine Size")
X = df[['engine-size']]
Y = df['price']
lm.fit(X, Y)
lm.predict(X)
print("Simple Linear Intercept "+str(lm.intercept_))
print("Simple Linear slope "+str(lm.coef_))

# To evaulate the best model (Simple Linear Regression) to use, visualization can be done with seaborn
# Regression plot - Shows reg line and scatter plot of data
width = 12
height = 10
plt.figure(figsize=(width, height))
ax = sns.regplot(x="highway-mpg", y="price", data=df)
ax.set_title("Highway-mpg vs Price Regression")
plt.ylim(0,)
plt.show()

plt.figure(figsize=(width, height))
ax1 = sns.regplot(x="peak-rpm", y="price", data=df)
ax1.set_title("Peak-rpm vs Price Regression")
plt.ylim(0,)
plt.show()

# I can show this numerically with pearsons coff
print("\nHighway-mpg vs Price correlation")
print(df['highway-mpg'].corr(df['price']))
print("\nPeak-rpm vs Price correlation")
print(df['peak-rpm'].corr(df['price']))

# A good way to visualize the variance of the data is a residual plot
# This plot shows the distance from the data point to the regression line - Variance of data
# Residuals on y and independent variable on the X
# If the residuals are randomly spread out a linear model is good
# If not and there is evidence of curvature use a none linear model
width = 12
height = 10
plt.figure(figsize=(width, height))
ax2 = sns.residplot(df['highway-mpg'], df['price'])
ax2.set_title("Residual plot of highway-mpg and price")
plt.show()

# Multiple Linear Regression - Relantionship between
# Multiple Predictor/independent variables - X
# Pedicted/dependent variable - Y
# in format Y = m1x1 + m2x2 + m(n)x(n) + b
# From corr we know Horsepower, curb-weight, engine-size and highway-mpg are good predictors
print("\nHorsepower, curb-weight, engine-size and highway-mpg")
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
lm.fit(Z, Y)
lm.predict(Z)
print("Multiple Linear Intercept "+str(lm.intercept_))
print("Multiple Linear slopes "+str(lm.coef_))

# Using normalizaed-losses and highway-mpg
print("\nNormalized-loses, highway-mpg")
Z = df[['normalized-losses', 'highway-mpg']]
lm.fit(Z, Y)
lm.predict(Z)
print("Multiple Linear Intercept "+str(lm.intercept_))
print("Multiple Linear slopes "+str(lm.coef_))


# For multiple linear regression a distribution plot can be used
# In this ex, it can be seen there is some correlation, but can be a lot better
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
lm.fit(Z, Y)
multiple_lin_reg = lm.predict(Z)
plt.figure(figsize=(width, height))

ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(multiple_lin_reg, hist=False, color="b", label="Fitted Values", ax=ax1)

plt.title('Actual vs Fitted Values for Price - Multiple Linear Regression')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

plt.show()


# Simple Polynomial regression
X = df['highway-mpg']
Y = df['price']
# Fitting the polynomial model
f = np.polyfit(X, Y, 6)
# Creating the polynomial equation
p = np.poly1d(f)
print(p)
# Plotting relationship
PlotPolly(p, X, Y, 'highway-mpg')

# Multivariate Polynomial Reg - Manual
pr = PolynomialFeatures(degree=2)
# Fits and transforms the data to be used
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
Z_pr = pr.fit_transform(Z)
lm.fit(Z_pr, df['price'])
# Makes the prediction and observes it through a distrubution plot
predict = lm.predict(Z_pr)
ax = sns.distplot(a=df['price'], label= "Actual Price", color='r', hist=False)
sns.distplot(a=predict, label="Predicted Price", color='b', hist=False,ax=ax)\
    .set_title("Multi-Price Poly reg Dist Plot - Degree: "+str(pr.degree))
plt.show()

# Pipelines - Simplifies the entire process of processing the data
# Polynomial Pipeline
Input = [('scale', StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model', LinearRegression())]
pipe = Pipeline(Input)
# Normalize, transform and fit the data in one line
pipe.fit(Z, df['price'])
# Normalize, transform and make a prediction in one line
predict = pipe.predict(Z)


# (Multi) Linear Pipeline
Input = [('scale', StandardScaler()), ('model', LinearRegression())]
pipe = Pipeline(Input)
pipe.fit(Z, df['price'])
predict2 = pipe.predict(Z)

ax = sns.distplot(a=df['price'], label= "Actual Price", color='r', hist=False)
sns.distplot(a=predict2, label="Predicted Price", color='b', hist=False,ax=ax)\
    .set_title("Multi-Price Dist Plot Pipeline Version")
plt.show()

# Quantifing model evaluation
# Can use R^2 and Mean Squared Error (MSE)
# R^2: measure to indicate how close the data is to the fitted regression line.
# The value of the R-squared is the percentage of variation of the response variable (y)
# that is explained by a linear model.
# MSE - measures the average of the squares of errors - difference between actual value(y) and the estimated value (ŷ)

# First model - SLR of HighwayMPG and Price
X = df[['highway-mpg']]
Y = df['price']
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]

lm.fit(X, Y)
SLR_prediction = lm.predict(X)
print('\nSLR - The R-square is: ', lm.score(X, Y))
mse = mean_squared_error(Y, SLR_prediction)
print('SLR - The mean square error of price and predicted value is: ', mse)

# Second Model - MLR of horsepower, curb-weight, engine-size, highway-mpg
lm.fit(Z, Y)
MLR_prediction = lm.predict(Z)
print('\nMLR - The R-square is: ', lm.score(Z, Y))
mse = mean_squared_error(Y, MLR_prediction)
print('MLR - The mean square error of price and predicted value is: ', mse)

# Third Model - Simple poly reg of highway mpg and price
f = np.polyfit(df['highway-mpg'], Y, 3)
p = np.poly1d(f)
r_squared = r2_score(Y, p(X))
print('\nSimple Poly - The R-square value is: ', r_squared)
mse = mean_squared_error(Y, p(X))
print('Simple Poly - The mean square error of price and predicted value is: ', mse)

# Fourth Model - Multi poly reg of horsepower, curb-weight, engine-size, highway-mpg
Input = [('scale', StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)),
         ('model', LinearRegression())]
pipe = Pipeline(Input)
pipe.fit(Z, df['price'])
predict = pipe.predict(Z)
r_squared = r2_score(Y, predict)
print('\nMultiple Poly - The R-square value is: ', r_squared)
mse = mean_squared_error(Y, predict)
print('Multiple Poly - The mean square error of price and predicted value is: ', mse)

# When comparing models, the model with the higher R-squared value is a better fit for the data
# When comparing models, the model with the smallest MSE value is a better fit for the data.

# After evaulating the models, prediction for future values can be made

# SLR/MLR
# Creating the range of new values
range_of_new_values = np.arange(1, 100, 1).reshape(-1, 1)
lm.fit(X, Y)
new_values = lm.predict(range_of_new_values)
plt.plot(range_of_new_values, new_values)
plt.show()

# Simple Poly
f = np.polyfit(df['highway-mpg'], Y, 3)
p = np.poly1d(f)
new_values = p(range_of_new_values)
plt.plot(range_of_new_values, new_values)
plt.show()


# Multi Poly
range_of_new_values = np.arange(1, 100, 1).reshape(-1, 1)
Input = [('scale', StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)),
         ('model', LinearRegression())]
pipe = Pipeline(Input)
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
pipe.fit(Z, df['price'])
predict = pipe.predict(Z)
new_values = predict[range_of_new_values]
plt.plot(range_of_new_values, new_values)
plt.show()

MPlotPolly(predict,range_of_new_values, new_values," Test")