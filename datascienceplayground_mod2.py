import pandas as pd
import numpy as np
import matplotlib as plt
from matplotlib import pyplot

#         Getting Data and Setting Data
url = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/auto.csv"
df = pd.read_csv(url, header = None)
df.replace('?',np.nan,inplace = True)

#path = "C:\\Users\\kobya\\Desktop\\auto.csv"
#df.to_csv(path)
#df = pd.read_csv(path)

#         Formatting Data
headers = ["symboling","normalized-losses","make","fuel","aspiration","num-of-doors",
           "body","drive-wheels","engine-location","wheel-base"
           ,"length","width","height","curb-weight","engine-type",
           "num-of-cylinders", "engine-size","fuel-system","bore","stroke",
           "compession-ratio","horsepower","peak-rpm","city-mpg","highway-mpg","price"]

df.columns = headers


#        To fix the datatype of columns
# df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
# df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
# df[["price"]] = df[["price"]].astype("float")
# df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")
# df["horsepower"]=df["horsepower"].astype(int, copy=True)

# This is how to drop rows/columns that have missing values. Axis = 0 means drop the col, axis = 1 means drop row
# can also add inplace = 'true' arguement to do it in 1 line
df = df.dropna(subset=["price"], axis=0)
# df.reset_index(drop=True, inplace=True) - to fix the index since rows were dropped
# or df = df.reset_index(drop=True)

# mean = df["columnName"].mean() - how to get the mean of a column - if typeerror df["columnName"].astype("float").mean()
# df["columnname"].replace(np.nan, mean) - replaces the first value with the second

#        To rename a column
# df.rename(columns={'"highway-mpg"':'highway-L/100km'},inplace = True)
# math functions can be performed on the cols (series)
# df["city-mpg"] = 235/df["city-mpg"]


# df.astype('int32').dtypes = changing data type of entire df
# df.astype({'col1': 'int32'}).dtypes = changing for a certain column


#               Viewing data
#pd.set_option("display.max_rows", None, "display.max_columns", None) - to display all
print(df.head(5))
print(df.tail(5))
print(df.dtypes)
print(df.describe(include='all'))
print(df.info)
print(df[['length','compession-ratio']].describe())
print(df['bore'])
print(df[['bore','length']].head(10))
#df['length'] = df['length']+5



#              Dataframe Normalization
# Done to make sure data weighting is correct and allows for accurate data anayasis - Can be done on a number of cols

#Simple feature scaling - series = series/ max value in series (makes values range from 0 to 1)
#df["columnname"] = df["columnname"]/df["columnname"].max()

#Min max method - series = (series - min value)/(max value - min value) (makes values range form 0 to 1)
#df["columnname"] = (df["columnname"]-df["columnname"].min())/(df["columnname"].max()-df["columnname"].min())

#Using Zscore - series = (series - average value)/standard deviation of series (typically range between -3 to +3)
#df["columnname"] = (df["columnname"]-df["columnname"].mean())/df["columnname"].std()




#           Binning - Making ranges of numbers - EX: 0 to 5 = low. 6 to 10 = mid, etc
# bins = np.linspace(df["columnName"].min(),df["columnName"].max(),4) - returns an array of 4 numbers evenly spaced out in the series
# groupNames = ["Low","Mid","High"]
# df["Price-Binned"] = pd.cut(df["price"], bins, labels = groupNames, include_lowest = True)

#           To visualize binning as a hisotgram

#df["horsepower"] = df["horsepower"].replace(np.nan, df["horsepower"].astype("float").mean())
#df["horsepower"]=df["horsepower"].astype(int, copy=True)
#bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
#group_names = ['Low', 'Medium', 'High']
#df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )

#pyplot.bar(group_names, df["horsepower-binned"].value_counts())

# set x/y labels and plot title
#plt.pyplot.xlabel("horsepower")
#plt.pyplot.ylabel("count")
#plt.pyplot.title("horsepower bins")
#pyplot.show()


#           To visualize them with values
#plt.pyplot.hist(df["horsepower"], bins = 3)

# set x/y labels and plot title
#plt.pyplot.xlabel("horsepower")
#plt.pyplot.ylabel("count")
#plt.pyplot.title("horsepower bins")
#pyplot.show()

#         To quantify objective data - dummy vaiables
#dummy = pd.get_dummies(df["fuel"]) - changes from strings to 0/1

#               Example of dummy
#dummy_variable_1 = pd.get_dummies(df["fuel-type"])
#dummy_variable_1.rename(columns={'fuel-type-diesel':'gas', 'fuel-type-diesel':'diesel'}, inplace=True)
# merge data frame "df" and "dummy_variable_1"
#df = pd.concat([df, dummy_variable_1], axis=1)

# drop original column "fuel-type" from "df"
#df.drop("fuel-type", axis = 1, inplace=True)



#         To count values in a series use df["columnName"].value_count() - very helpful
#         Can do df["columnName"].value_count().idxmax() - find most frequent value