import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# Function to plot a dist plot
def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)
    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')

    plt.show()
    plt.close()


# Updated Polynomial plot function
def PollyPlot(xtrain, xtest, y_train, y_test, lr, poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    # training data
    # testing data
    # lr:  linear regression object
    # poly_transform:  polynomial transformation object

    xmax = max([xtrain.values.max(), xtest.values.max()])

    xmin = min([xtrain.values.min(), xtest.values.min()])

    x = np.arange(xmin, xmax, 0.1)

    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    plt.close()

# Allows for testing of different orders of polynomials and test data seperations
def f(order, test_data):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_data, random_state=0)
    pr = PolynomialFeatures(degree=order)
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    x_test_pr = pr.fit_transform(x_test[['horsepower']])
    poly = LinearRegression()
    poly.fit(x_train_pr, y_train)
    PollyPlot(x_train[['horsepower']], x_test[['horsepower']], y_train, y_test, poly, pr)


path = 'module_5_auto.csv'
df = pd.read_csv(path)

# Setting the DF to only numeric values as I'm not using the rest
df = df._get_numeric_data()

#                     Manual train/test
# For model eval I need to split into test and train data
# This is done to see how well the model predicts unknown values from known values...
# The predicted "unknowns" can then be compared to the test sample to see how accurate the model is

# If model only evals good on train data means its fit to the noise (random variance) of the model
# If model only evals good on test data means its more fit to the relantionship but not fit to the given data

# First step is seperating the price values from the main dataset
y_data = df['price']
x_data = df.drop('price',axis=1)

# Setting of the training and testing data
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=1)
print("15% test version")
print("number of test samples :", x_test.shape[0])
print("number of training samples:", x_train.shape[0])

# My example
print("\n40% test version")
x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(x_data, y_data, test_size=.4,random_state=0)
print("number of test samples :", x_test_1.shape[0])
print("number of training samples:", x_train_1.shape[0])

print("\n")

# Model creation and fitting
linear_reg = LinearRegression()
linear_reg.fit(x_train[['horsepower']], y_train)
# Calculating the R^2 of the model using the test data
print("Score of test data - 15%: "+str(linear_reg.score(x_test[['horsepower']], y_test)))
# Calculating the R^2 of the model using the test data
print("Score of train data - 15%: "+str(linear_reg.score(x_train[['horsepower']], y_train)))

print("\n")

# My example using 10% test data
x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(x_data, y_data, test_size=0.1, random_state=0)
linear_reg.fit(x_train_2[['horsepower']], y_train_2)
print("Score of test data - 10%: "+str(linear_reg.score(x_test[['horsepower']], y_test)))
print("Score of train data - 10%: "+str(linear_reg.score(x_train[['horsepower']], y_train)))


#               Cross validation testing/score

# This creates N equal partitions of the data set and uses one of them for testing and the rest for training
# Then does the same for each partition
# Gives the average of scores to give a more accurate view of model accuracy
cross_val_R = cross_val_score(linear_reg, x_data[['horsepower']], y_data, cv=4)
print("\n")
print("The mean of the folds are", cross_val_R.mean(), "and the standard deviation is", cross_val_R.std())
print(cross_val_R)

# Different scoring methods can be used such as MSE
cross_val_MSE = -1 * cross_val_score(linear_reg,x_data[['horsepower']], y_data, cv=4, scoring='neg_mean_squared_error')
print("\n")
print("The mean of the folds are", cross_val_MSE.mean(), "and the standard deviation is", cross_val_MSE.std())
print(cross_val_MSE)

# My Example
cross_val_MSE_2 = -1 * cross_val_score(linear_reg,x_data[['horsepower']], y_data, cv=2, scoring='neg_mean_squared_error')
print("\n")
print("The mean of the folds are", cross_val_MSE_2.mean(), "and the standard deviation is", cross_val_MSE_2.std())
print(cross_val_MSE_2)


#                   Overfitting, Underfitting and Model Selection
# Fitting and creating prediction
multi_lin_reg = LinearRegression()
multi_lin_reg.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_train)
# Prediction using train data
mlr_train_predict = multi_lin_reg.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
# Prediction using test data
mlr_test_predict = multi_lin_reg.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])

# When using the test model the accuracy of the model drops although both of them were trained with the same sample
# Figure 1: Plot of predicted values using the training data compared to the training data.
Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'
DistributionPlot(y_train, mlr_train_predict, "Actual Values (Train)", "Predicted Values (Train)", Title)

# Figure 2: Plot of predicted value using the test data compared to the test data.
Title_2 = 'Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
DistributionPlot(y_test, mlr_test_predict, "Actual Values (Test)", "Predicted Values (Test)", Title_2)


# Overfitting occurs when the model fits the noise (random variance) of the values not the relantionship
# Therefore when testing your model using the test-set, the model does not perform as well as it is modelling noise
# Due to the fact of there being less samples in the testing sample?

# Creating a polynomial model
x_train_3, x_test_3, y_train_3, y_test_3 = train_test_split(x_data, y_data, test_size=0.45, random_state=0)
pr = PolynomialFeatures(degree=5)
x_train_poly = pr.fit_transform(x_train_3[['horsepower']])
x_test_poly = pr.fit_transform(x_test_3[['horsepower']])

poly = LinearRegression()
poly.fit(x_train_poly, y_train_3)
poly_predict = poly.predict(x_test_poly)

# Comparing values of prediction
print("\nPredicted values:", poly_predict[0:4])
print("True values:", y_test[0:4].values)

# Visualizing it and eval it
PollyPlot(x_train_3[['horsepower']], x_test_3[['horsepower']], y_train_3, y_test_3, poly, pr)
print("\nR sqrt of training data:", poly.score(x_train_poly, y_train_3))
print("R sqrt of testing data:", poly.score(x_test_poly, y_test_3))

# To determine the best polynomial order the below can be done
Rsqu_test = []
order = [*range(1, 6, 1)]
for n in order:
    pr = PolynomialFeatures(degree=n)

    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    x_test_pr = pr.fit_transform(x_test[['horsepower']])

    poly.fit(x_train_pr, y_train)

    Rsqu_test.append(poly.score(x_test_pr, y_test))

plt.plot(order, Rsqu_test)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using Test Data')
plt.text(3, 0.75, 'Maximum R^2 ')
plt.show()

# Questions - Multiple poly regression
pr2 = PolynomialFeatures(degree=2)
x_train_poly2 = pr2.fit_transform(x_train_3[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
x_test_poly2 = pr2.fit_transform(x_test_3[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
poly2 = LinearRegression()
poly2.fit(x_train_poly2, y_train_3)

poly2_predict = poly2.predict(x_test_poly2)
Title = 'Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data - MPR'
DistributionPlot(y_test_3, poly2_predict, "Actual Values (Test)", "Predicted Values (Test)", Title)

# Ridge Regression
# Scales the polynomial coeffs
RidgeModel = Ridge(alpha=0.1)
RidgeModel.fit(x_train_poly2, y_train_3)
ridge_predict = RidgeModel.predict(x_test_poly2)

# Comparing to actual values
print('\npredicted:', ridge_predict[0:4])
print('test set :', y_test_3[0:4].values)

# To select a value of alpha that decreases the test error a loop can be used
poly3 = PolynomialFeatures(degree=2)
x_train_poly_3 = poly3.fit_transform(x_train_3[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg', 'normalized-losses','symboling']])
x_test_poly_3 = poly3.fit_transform(x_test_3[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg', 'normalized-losses','symboling']])
Rsqu_test = []
Rsqu_train = []
ALFA = 10 * np.array(range(0, 1000))
for alfa in ALFA:
    RidgeModel = Ridge(alpha=alfa)
    RidgeModel.fit(x_train_poly_3, y_train_3)
    Rsqu_test.append(RidgeModel.score(x_test_poly_3, y_test_3))
    Rsqu_train.append(RidgeModel.score(x_train_poly_3, y_train_3))

# Plotting the R sqrt for alpha vals
width = 12
height = 10
plt.figure(figsize=(width, height))

# Figure 6:The blue line represents the R^2 of the test data, and the red line represents the R^2 of the training data.
# The x-axis represents the different values of Alfa
# The red line in figure 6 represents the R^2 of the test data, as Alpha increases the R^2 decreases;
# therefore as Alfa increases the model performs worse on the test data.
# The blue line represents the R^2 on the validation data, as the value for Alfa increases the R^2 decreases.
plt.plot(ALFA, Rsqu_test, label='validation data  ')
plt.plot(ALFA, Rsqu_train, 'r', label='training Data ')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.legend()
plt.show()

# My exmaple
RidgeModel2 = Ridge(alpha=10)
RidgeModel2.fit(x_train_poly_3, y_train_3)
print("\n", RidgeModel2.score(x_test_poly_3, y_test))

# Grid search - Finds best based on combos of hyperparameters
# set hyper parameters to choose from
hyperparameters = [{'alpha': [0.001,0.1,1, 10, 100, 1000, 10000, 100000, 100000],'normalize':[True, False]}]
# Create Ridge reg object
RR = Ridge()
# Create grid search object
Grid = GridSearchCV(RR, hyperparameters,cv=4)
# Fit the grid search
Grid.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)
# Finds best model combo and prints
BestRR=Grid.best_estimator_
print(BestRR)
# Lastly test the data on test data
print(BestRR.score(x_test_3[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_test_3))
