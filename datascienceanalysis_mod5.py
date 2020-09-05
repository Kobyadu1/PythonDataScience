import pandas as pd
from sklearn import model_selection as ms
from sklearn.model_selection import cross_val_score as cvs
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
path='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
df = pd.read_csv(path)


# In model devlopment/evaluation sometimes the training data can be used to test as well
lm = LinearRegression()
X = df[['highway-mpg']]
Y = df['price']
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
x_train, x_test, y_train, y_test = ms.train_test_split(Z,Y,train_size=.7, random_state=0)

# To get a more realistic picture we can use cross examination to evaluate the model more accuratly
# First is a regression object, the data, and last parameter is the amount of partions of the data
scores = cvs(lm, Z, df['price'], cv=3)
print(scores.mean())

# How to choose polynomial order
# Under fit is when model is too simple for the given data
# Where as over fit is when the model is really good at lining up with model but terrible at predicting new data
# When using a poly order that is too high over fit can happen and when using a linear model or a lower level polynomial
# Under fit can occur

# Theres many ways to visualize the best order
# An exmaple is graphing/printing the R-Squared or MSE

R_array = list()
for n in range(1, 5):
    pr = PolynomialFeatures(degree=n)
    x_train_pr = pr.fit_transform(x_train)
    x_test_pr = pr.fit_transform(x_test)
    lm.fit(x_train_pr, y_train)
    R_array.append(lm.score(x_test_pr, y_test))
print(R_array)

# Ridge regression makes it so that the values of the coeff are not too big, adjusting the model for over/under fit
# Alpha value has to be correct tho so same method from above can be applied
# Only really for higher degree values that start overfitting the model
RidgeModel = Ridge(alpha=.1)
RidgeModel.fit(Z, Y)
prediction = RidgeModel.predict(Z)

# Grid search takes model and objects I would like to train and returns the best model based on R^2 or MSE
hyper_para = [{'alpha': [.001, .1, 1, 10, 100], 'normalize': [True, False]}]
ridge_reg = Ridge()

Grid1 = GridSearchCV(ridge_reg, hyper_para, cv=4)
Grid1.fit(Z, Y)
print(Grid1.best_estimator_)
scores = Grid1.cv_results_
print("\n")
print(scores['mean_test_score'])

