import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
from matplotlib import pyplot
from scipy import stats

#         Getting Data and Setting Data
url = "clean_auto.csv"
df = pd.read_csv(url)

print(df['bore'].describe(include="all"))
print(df.describe())
print(df['bore'].value_counts())


drive_wheel_count = df['drive-wheels'].value_counts()
drive_wheel_count.rename({'drive-wheels':'value-counts'}, inplace = True)
drive_wheel_count.index.name = 'drive-wheels'

# Boxplot shows outliers and middle 50 and quartiles
sns.boxplot(x='drive-wheels', y = 'price', data=df)
pyplot.show()

# Scatter plot is good for determining correlation
# X value is predictor while the value you are trying to prediect is the y
y_value = df['price']
x_value = df['engine-size']
pyplot.scatter(x_value, y_value)
pyplot.show()

# GroupBY - Groups data with price avergae shown in 3rd column
df_test = df[['drive-wheels', 'body-style', 'price']]
df_grp = df_test.groupby(['drive-wheels','body-style'],as_index = False,).mean().sort_values(by=['price'])
print(df_grp)

# Pivot Table to make it easier to read - rectangular format
df_pivot = df_grp.pivot(index='drive-wheels',columns='body-style')
print(df_pivot)

#Plotting as heatmap
sns.heatmap(df_pivot,cmap="RdBu")
pyplot.show()

# Creating a regression line
sns.regplot(x='engine-size',y='price', data=df)
pyplot.ylim(0)
pyplot.show()

# Correlation stats
pearson, p_value = stats.pearsonr(df['engine-size'], df['price'])
print(str(pearson)+" "+str(p_value))

# Correlation Matrix - Shows how each variable correlates with each other - Pearson Coff
corrMatrix = df.corr()
print (corrMatrix)
# sns.heatmap(corrMatrix, annot=True)
sns.heatmap(corrMatrix,cmap="RdBu")
pyplot.show()

# ANOVA
sns.barplot(x='make', y='price', data=df)
pyplot.show()
df_anova = df[['make','price']]
group_anova = df_anova.groupby(['make'])
anova_results = stats.f_oneway(group_anova.get_group('honda')['price'],group_anova.get_group('subaru')['price'])
anova_results2 = stats.f_oneway(group_anova.get_group('honda')['price'],group_anova.get_group('jaguar')['price'])
print(anova_results+" "+anova_results2)
# When F score above one and P score below .05 theres a strong correlation
