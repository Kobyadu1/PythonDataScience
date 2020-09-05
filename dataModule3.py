import pandas as pd
from matplotlib import pyplot
from scipy import stats
import seaborn as sns

# After cleaning data, use the tools here to determine which datapoints are valid for model devlopment

path='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
df = pd.read_csv(path)

# Find correlation between int64 or float64 dtypes
print(df[['bore','stroke','compression-ratio','horsepower']].corr(method="pearson"))

# Engine size as potential predictor variable of price - Looking at correlation
sns.regplot(x="engine-size", y="price", data=df)
pyplot.ylim(0,)
pyplot.show()
print(df[["engine-size", "price"]].corr())
print()

# Highway mpg is a potential predictor variable of price
sns.regplot(x="highway-mpg", y="price", data=df)
pyplot.ylim(0,)
pyplot.show()
print(df[["highway-mpg", "price"]].corr())
print()

# Peak RPM is a week predictor
sns.regplot(x="peak-rpm", y="price", data=df)
pyplot.ylim(0,)
pyplot.show()
print(df[['peak-rpm','price']].corr())
print()

# Low pearson coff, so expected to have a low correlation
print(df[["stroke","price"]].corr())
sns.regplot(x='stroke',y='price',data=df)
pyplot.ylim(0,)
pyplot.show()

# Comparing categorical data

# Significant overlap between the boxplots, so not a good predictor
sns.boxplot(x="body-style", y="price", data=df)
pyplot.show()

# No overlap, could be a predictor
sns.boxplot(x="engine-location", y="price", data=df)
pyplot.show()

# Almost no overlap, could be a predictor
sns.boxplot(x="drive-wheels", y="price", data=df)
pyplot.show()

# Describe method
# shows the following
# the count of that variable
# the mean
# the standard deviation (std)
# the minimum value
# the IQR (Interquartile Range: 25%, 50% and 75%)
# the maximum value
print(df.describe())

# To have it so it does not skip objects (string)
print(df.describe(include=['object']))

# Shows how many units of each characteristic/variable - only for a series not a DF so single [] not [[]]
print(df['drive-wheels'].value_counts())

# Can convert series to dataframe by
drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
# changes the name of the index column - first column in this case
drive_wheels_counts.index.name = 'drive-wheels'
print(drive_wheels_counts)

# engine-location as variable - would not be a good predictor as not much variance
engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
print(engine_loc_counts)

# Finds the unique values within the series
print(df['drive-wheels'].unique())

# Groups - The data is grouped based on one or several variables and analysis is performed on the individual groups.
df_group_one = df[['drive-wheels', 'body-style', 'price']]
# grouping results - We can then calculate the average price for each of the different categories of data.
df_group_one = df_group_one.groupby(['drive-wheels'],as_index=False).mean()
print(df_group_one)

# Can also group by multiple variables
df_gptest = df[['drive-wheels','body-style','price']]
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()
print(grouped_test1)

# After data has been grouped by multiple variables
# It is easier to read when pivoted
grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')
print(grouped_pivot)

# Often, we won't have data for some of the pivot cells.
# We can fill these missing cells with the value 0,
# but any other value could potentially be used as well.
# It should be mentioned that missing data is quite a complex subject and is an entire course on its own.
grouped_pivot = grouped_pivot.fillna(0)
print(grouped_pivot)

# My example
body_style_group = df[['body-style','price']].groupby(['body-style'],as_index = False).mean()
print(body_style_group)

# Heatmap of grouped
sns.heatmap(data=grouped_pivot,cmap='RdBu',annot=True)
pyplot.show()

# Correlation
# Pearson corr from -1 to 1 measures linear correlation
# 1: Total positive linear correlation.
# 0: No linear correlation, the two variables most likely do not affect each other.
# -1: Total negative linear correlation.
print(df.corr(method="pearson"))

# P-Value
# P-value is the probability value that the correlation between these two variables is statistically significant.
# Normally, we choose a significance level of 0.05
# means that we are 95% confident that the correlation between the variables is significant
# p-value is  <  0.001: we say there is strong evidence that the correlation is significant.
# the p-value is  <  0.05: there is moderate evidence that the correlation is significant.
# the p-value is  <  0.1: there is weak evidence that the correlation is significant.
# the p-value is  >  0.1: there is no evidence that the correlation is significant.

# Since the p-value is  <  0.001, the correlation between wheel-base and price is statistically significant
# although the linear relationship isn't extremely strong (~0.585)
pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

# Since the p-value is  <  0.001, the correlation between horsepower and price is statistically significant
# and the linear relationship is quite strong (~0.809, close to 1)
pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)

# The Analysis of Variance (ANOVA) is a statistical method used to test
# whether there are significant differences between the means of two or more groups.
# ANOVA returns two parameters:

# F-test score: ANOVA assumes the means of all groups are the same,
# calculates how much the actual means deviate from the assumption, and reports it as the F-test score.
# A larger score means there is a larger difference between the means.
# P-value: P-value tells how statistically significant is our calculated score value.

# If our price variable is strongly correlated with the variable we are analyzing
# expect ANOVA to return a sizeable F-test score and a small p-value.

# First group the data
grouped_test2=df_gptest[['drive-wheels', 'price']].groupby(['drive-wheels'])
print(grouped_test2.head(2))

# To get a group from a groupby df
print(grouped_test2.get_group('4wd')['price'])

# ANOVA - To see how correlated the groups are with each other
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'],
                              grouped_test2.get_group('4wd')['price'])
print("ANOVA results: F=", f_val, ", P =", p_val)

# By using linear correlation we can get a good idea of which NUMERICAL columns are good for price prediction
# By using Anova we can get a good idea of which CATEGORICAL columns are good for price prediction


# In this case
# Continuous numerical variables:
# Length
# Width
# Curb-weight
# Engine-size
# Horsepower
# City-mpg
# Highway-mpg
# Wheel-base
# Bore

# Categorical variables:
# Drive-wheels
