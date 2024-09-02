'''CRISP-ML(Q):
    
Business Problem:
A glass manufacturing plant uses different earth elements to design new glass materials based on customer requirements. For that, they would like to automate the process of classification as it’s a tedious job to manually classify them. Help the company achieve its objective by correctly classifying the glass type based on the other features.

Business Objective: Maximize Glasstype Detection
Business Constraints: Minimize labour Cost & Maximize productivity

Success Criteria: 
Business success criteria: Reduce the time consumption for glass type detection atleast by 50 percent
Machine Learning success criteria: Achieve an accuracy of atleast 98%
Economic success criteria: Reducing labour cost atleast by 20%

Data Collection:
The features of the collected data are
RI	Refractive Index
Na	Sodium Content
Mg	Magnesium Content
Al	Aluminum Content
Si	Silicon Content
K	Potassium Content
Ca	Calcium Content
Ba	Barium Content
Fe	Iron Content
Type	Type of Glass

'''    
    
# CODE MODULARITY IS EXTREMELY IMPORTANT
# Import the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

# pip install sklearn_pandas

from sklearn.pipeline import Pipeline
from sklearn_pandas import DataFrameMapper
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import sklearn.metrics as skmet
import pickle


# MySQL Database connection

from sqlalchemy import create_engine, text

glassdata = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/Data Science/KNN_Classifier/Assignments/KNN/glass.csv")

# Creating engine which connect to MySQL
user = 'root' # user name
pw = '1234' # password
db = 'glass_db' # database

# creating engine to connect database
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# dumping data into database 
glassdata.to_sql('glass', con = engine, if_exists = 'replace', chunksize = 1000, index = False)



# loading data from database
sql = 'select * from glass'

glassdf = pd.read_sql_query(text(sql), con = engine.connect())

print(glassdf)

# Data Preprocessing & EDA
glassdf.info()   # No missing values observed

a = glassdf.describe()

# Seperating input and output variables 
glassdf_X = pd.DataFrame(glassdf.iloc[:, :9])
glassdf_y = pd.DataFrame(glassdf.iloc[:, 9])

# glassdf_y = glassdf_y.astype(str)
# glassdf_y.info()

# EDA and Data Preparation
glassdf_X.info()


import joblib

# Define scaling pipeline
scale_pipeline = Pipeline([('scale', MinMaxScaler())])

preprocess_pipeline2 = ColumnTransformer([('scale', scale_pipeline, glassdf_X.columns)]) 

processed2 = preprocess_pipeline2.fit(glassdf_X)
processed2

# Save the Scaling pipeline
joblib.dump(processed2, 'processed2')

import os 
os.getcwd()

# Normalized data frame (considering the numerical part of data)

glassclean_n = pd.DataFrame(processed2.transform(glassdf_X), columns = glassdf_X.columns)

res = glassclean_n.describe()
res

# Separating the input and output from the dataset
# X = np.array(glassclean_n.iloc[:, :]) # Predictors 
Y = np.array(glassdf_y['Type']) # Target

X_train, X_test, Y_train, Y_test = train_test_split(glassclean_n, Y,
                                                    test_size = 0.2, random_state = 0)

X_train.shape
X_test.shape

x_train = np.array(X_train)
x_test = np.array(X_test)

# Model building
knn = KNeighborsClassifier(n_neighbors = 11)

KNN = knn.fit(X_train, Y_train)  # Train the kNN model

# Evaluate the model with train data
pred_train = knn.predict(x_train)  # Predict on train data

pred_train

# Cross table
pd.crosstab(Y_train, pred_train, rownames = ['Actual'], colnames = ['Predictions']) 

print(skmet.accuracy_score(Y_train, pred_train))  # Accuracy measure

import numpy as np

print(np.mean(Y_train == pred_train))

# Predict the class on test data
pred = knn.predict(x_test)
pred

# Evaluate the model with test data
print(skmet.accuracy_score(Y_test, pred))

pd.crosstab(Y_test, pred, rownames = ['Actual'], colnames = ['Predictions']) 

cm = skmet.confusion_matrix(Y_test, pred)

cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix = cm)
cmplot.plot()
cmplot.ax_.set(title = 'glass Detection - Confusion Matrix', 
               xlabel = 'Predicted Value', ylabel = 'Actual Value')


# creating empty list variable 
acc = []

# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# storing the accuracy values

for i in range(1, 15, 2):
    # print(i)
    neigh = KNeighborsClassifier(n_neighbors = i)
    neigh.fit(x_train, Y_train)
    train_acc = np.mean(neigh.predict(x_train) == Y_train)
    test_acc = np.mean(neigh.predict(x_test) == Y_test)
    diff = train_acc - test_acc
    acc.append([diff, train_acc, test_acc])
    
acc
    
# Plotting the data accuracies
plt.plot(np.arange(1, 15, 2), [i[1] for i in acc], "ro-")
plt.plot(np.arange(1, 15, 2), [i[2] for i in acc], "bo-")




# Hyperparameter optimization
from sklearn.model_selection import GridSearchCV

k_range = list(range(1, 21, 2))
param_grid = dict(n_neighbors = k_range)
  
# Defining parameter range
grid = GridSearchCV(knn, param_grid, cv = 5, 
                    scoring = 'accuracy', 
                    return_train_score = False, verbose = 1)

help(GridSearchCV)

KNN_new = grid.fit(x_train, Y_train) 

print(KNN_new.best_params_)

accuracy = KNN_new.best_score_ *100
print("Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy) )

# Predict the class on test data
pred = KNN_new.predict(x_test)
pred

print(skmet.accuracy_score(Y_test, pred))


cm = skmet.confusion_matrix(Y_test, pred)

cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix = cm)
cmplot.plot()
cmplot.ax_.set(title = 'glass Detection - Confusion Matrix', 
               xlabel = 'Predicted Value', ylabel = 'Actual Value')

# Save the model
knn_best = KNN_new.best_estimator_
pickle.dump(knn_best, open('knn.pkl', 'wb'))

import os
os.getcwd()
