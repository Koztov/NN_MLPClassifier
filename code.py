#http://stackabuse.com/introduction-to-neural-networks-with-scikit-learn/

# -*- coding: utf-8 -*-
import pandas as pd

# Assign colum names to the dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', '_class']

# Read dataset to pandas dataframe
irisdata = pd.read_csv("C:/Users/Koztov/Vscode Workspace/Python/NN_MLPClassifier/iris.csv", names=names)  

print(irisdata.head())  

# Assign data from first four columns to X variable
X = irisdata.iloc[:, 0:4]

# Assign data from first fifth columns to y variable
y = irisdata.select_dtypes(include=[object]) 
print(y._class.unique()) 

#convert to numerical values
from sklearn import preprocessing  
le = preprocessing.LabelEncoder()
y = y.apply(le.fit_transform)  
print(y._class.unique()) 

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  

from sklearn.neural_network import MLPClassifier  
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)  
mlp.fit(X_train, y_train.values.ravel()) 
predictions = mlp.predict(X_test)  

from sklearn.metrics import classification_report, confusion_matrix  
print("confusion_matrix:")
print(confusion_matrix(y_test,predictions))  
print("classification_report:")
print(classification_report(y_test,predictions))  