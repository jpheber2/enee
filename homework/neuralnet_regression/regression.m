# Project1 Week 2 - Linear Regression on California Housing Data
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
# 1. Access the California Housing data
# 2. Read the CSV data into python
names = pd.read_csv('/Users/Madeleine/Downloads/CaliforniaHousing/cal_housing.domain')
print (names)

# 3. Determine the number of features/attributes
data = pd.read_csv('/Users/Madeleine/Downloads/CaliforniaHousing/cal_housing.data',
	names = ['longitude', 'latitude', 'housingMedianAge', 'totalRooms', 	'totalBedrooms', 'population', 'households', 'medianIncome', 'medianHouseValue'])
print (data.head())

# Determine the number of houses in the database
# Determine the type of data collected for each attribute
print (data.info())

# Determine which attributed are not completly collected over all data types
# Determine range of data collected for each attribute
print (data.describe())

# 4. Make a histogram plot for each attribute
data.hist(bins=50, figsize=(20,15))

# import linear regression functions
import sklearn
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

# split data into prediction features and target results, x and y
X=data.drop('medianHouseValue', axis = 1)
Y=data.medianHouseValue

# 5. Apply linear regression using each feature and measure the error
lr.fit(X.longitude.values.reshape(-1,1), Y)
print ("Variance error when applying linear regressing using 'longitude' feature = {0:.3f}%" .format((1-lr.score(X.longitude.values.reshape(-1,1), Y))*100))
lr.fit(X.latitude.values.reshape(-1,1), Y)
print ("Variance error when applying linear regressing using 'latitude' feature = {0:.3f}%" .format((1-lr.score(X.latitude.values.reshape(-1,1), Y))*100))
lr.fit(X.housingMedianAge.values.reshape(-1,1), Y)
print ("Variance error when applying linear regressing using 'housingMedianAge' feature = {0:.3f}%" .format((1-lr.score(X.housingMedianAge.values.reshape(-1,1), Y))*100))
lr.fit(X.totalRooms.values.reshape(-1,1), Y)
print ("Variance error when applying linear regressing using 'totalRooms' feature = {0:.3f}%" .format((1-lr.score(X.totalRooms.values.reshape(-1,1), Y))*100))
lr.fit(X.totalBedrooms.values.reshape(-1,1), Y)
print ("Variance error when applying linear regressing using 'totalBedrooms' feature = {0:.3f}%" .format((1-lr.score(X.totalBedrooms.values.reshape(-1,1), Y))*100))
lr.fit(X.population.values.reshape(-1,1), Y)
print ("Variance error when applying linear regressing using 'population' feature = {0:.3f}%" .format((1-lr.score(X.population.values.reshape(-1,1), Y))*100))
lr.fit(X.households.values.reshape(-1,1), Y)
print ("Variance error when applying linear regressing using 'households' feature = {0:.3f}%" .format((1-lr.score(X.households.values.reshape(-1,1), Y))*100))
lr.fit(X.medianIncome.values.reshape(-1,1), Y)
print ("Variance error when applying linear regressing using 'medianIncome' feature = {0:.3f}%" .format((1-lr.score(X.medianIncome.values.reshape(-1,1), Y))*100))

# Choose 2 features with lowest error
X2 = X.filter(['medianIncome','latitude'], axis=1)

# 6. Apply linear regression on the 2 features from the previous step
from sklearn.model_selection import train_test_split
 
X_train, X_test, Y_train, Y_test = train_test_split( X2, Y, test_size=0.2, random_state = 42)
 
lr.fit(X_train, Y_train)
print ("Variance when applying linear regressing using 'medianIncome' and 'latitude' = {0:.3f}%" .format((1-lr.score(X_train, Y_train))*100))

# Project1 Week 2 - Logistic Regression on Iris Dataset
# import libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

# 7. Import dataset from sklearn and load iris dataset
from sklearn import datasets
iris = datasets.load_iris()

# 8. Explore the dataset
print (iris.DESCR)

print(iris.keys())

# number of samples
print (iris.target.shape)
# number of classes
print (iris.target_names.shape)
# number of features
print (iris.data.shape)
# type of data for each feature
print (iris.data.dtype)

# move dataset into x data and y target pandas dataframe
X = pd.DataFrame(iris.data, columns = iris.feature_names)
Y = pd.DataFrame(iris.target, columns = ['class'])

# 9. Split the data into two classes only
X2 = pd.DataFrame(X.values[:100,:], columns = iris.feature_names)
Y2 = pd.DataFrame(Y.values[:100,:], columns = ['class'])

# 10. Generate plots for data of two classes, comparing features
pd.plotting.scatter_matrix(X2, figsize=(12, 10))

# 11. Apply logistic regression and determine error for seperating two classes with two feature
logreg.fit(X2.filter(['sepal length (cm)','sepal width (cm)'], axis=1), Y2.values.ravel())
print ("Variance error when applying logistic on 'sepal length' 'sepal width' with two classes = {0:.0f}%" .format((1-logreg.score(X2.filter(['sepal length (cm)','sepal width (cm)'], axis=1), Y2))*100))
logreg.fit(X2.filter(['sepal length (cm)','petal length (cm)'], axis=1), Y2.values.ravel())
print ("Variance error when applying logistic on 'sepal length' 'petal length' with two classes = {0:.0f}%" .format((1-logreg.score(X2.filter(['sepal length (cm)','petal length (cm)'], axis=1), Y2))*100))
logreg.fit(X2.filter(['sepal length (cm)','petal width (cm)'], axis=1), Y2.values.ravel())
print ("Variance error when applying logistic on 'sepal length' 'petal width' with two classes = {0:.0f}%" .format((1-logreg.score(X2.filter(['sepal length (cm)','petal width (cm)'], axis=1), Y2))*100))
logreg.fit(X2.filter(['sepal width (cm)','petal length (cm)'], axis=1), Y2.values.ravel())
print ("Variance error when applying logistic on 'sepal width' 'petal length' with two classes = {0:.0f}%" .format((1-logreg.score(X2.filter(['sepal width (cm)','petal length (cm)'], axis=1), Y2))*100))
logreg.fit(X2.filter(['sepal width (cm)','petal width (cm)'], axis=1), Y2.values.ravel())
print ("Variance error when applying logistic on 'sepal width' 'petal width' with two classes = {0:.0f}%" .format((1-logreg.score(X2.filter(['sepal width (cm)','petal width (cm)'], axis=1), Y2))*100))
logreg.fit(X2.filter(['petal length (cm)','petal width (cm)'], axis=1), Y2.values.ravel())
print ("Variance error when applying logistic on 'petal length' 'petal width' with two classes = {0:.0f}%" .format((1-logreg.score(X2.filter(['petal length (cm)','petal width (cm)'], axis=1), Y2))*100))

# 12. Generate plots for data of all classes and compare features
pd.plotting.scatter_matrix(X, figsize=(12, 10))

# Apply logistic regression and determine error for seperating all classes with two features
logreg.fit(X.filter(['sepal length (cm)','sepal width (cm)'], axis=1), Y.values.ravel())
print ("Variance error when applying logistic on 'sepal length' 'sepal width' with all classes = {0:.0f}%" .format((1-logreg.score(X.filter(['sepal length (cm)','sepal width (cm)'], axis=1), Y))*100))
logreg.fit(X.filter(['sepal length (cm)','petal length (cm)'], axis=1), Y.values.ravel())
print ("Variance error when applying logistic on 'sepal length' 'petal length' with all classes = {0:.0f}%" .format((1-logreg.score(X.filter(['sepal length (cm)','petal length (cm)'], axis=1), Y))*100))
logreg.fit(X.filter(['sepal length (cm)','petal width (cm)'], axis=1), Y.values.ravel())
print ("Variance error when applying logistic on 'sepal length' 'petal width' with all classes = {0:.0f}%" .format((1-logreg.score(X.filter(['sepal length (cm)','petal width (cm)'], axis=1), Y))*100))
logreg.fit(X.filter(['sepal width (cm)','petal length (cm)'], axis=1), Y.values.ravel())
print ("Variance error when applying logistic on 'sepal width' 'petal length' with all classes = {0:.0f}%" .format((1-logreg.score(X.filter(['sepal width (cm)','petal length (cm)'], axis=1), Y))*100))
logreg.fit(X.filter(['sepal width (cm)','petal width (cm)'], axis=1), Y.values.ravel())
print ("Variance error when applying logistic on 'sepal width' 'petal width' with all classes = {0:.0f}%" .format((1-logreg.score(X.filter(['sepal width (cm)','petal width (cm)'], axis=1), Y))*100))
logreg.fit(X.filter(['petal length (cm)','petal width (cm)'], axis=1), Y.values.ravel())
print ("Variance error when applying logistic on 'petal length' 'petal width' with all classes = {0:.0f}%" .format((1-logreg.score(X.filter(['petal length (cm)','petal width (cm)'], axis=1), Y))*100))