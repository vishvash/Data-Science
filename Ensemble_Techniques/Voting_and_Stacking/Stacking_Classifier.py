'''
Business Understanding:
Business Problem: When scientists do research it is becoming extremely difficult 
to seggregate the '3' species - Versicolor, Virginica, Setosa. 

Business Objective: Maximize Species Detection Accuracy
Business Constraint: Minimize Cost of Detection

Success Criteria: 
Business - Increase effectiveness of species detection by at least 50%
ML - Achieve an accuracy of more than 80%
Economic - Save upto $1M annually

Data Understanding:
Data
a. Sepal length
b. Sepal width
c. Petal length
d. Petal width

Target 
e. Species (Versicolor, Virginica, Setosa)

150 observations & 5 columns
'''


from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
# from sklearn import metrics
import numpy as np
import pandas as pd
import pickle

# Load the dataset
iris = load_iris()


# Create the dataframe
df_features = pd.DataFrame(data = iris.data, columns = iris.feature_names)
print(df_features)

df_target = pd.DataFrame(data = iris.target, columns = ['species'])
print(df_target)

# Dataset
final = pd.concat([df_features, df_target], axis = 1)

# Segregate the data into Input and Output
X = np.array(final.iloc[:, :4]) # Predictors 
y = np.array(final['species']) # Target

# Split into Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify = y, random_state = 42)

X_train[0:5]

y_train[0:5]

# Base estimators
estimators = [('rf', RandomForestClassifier(n_estimators = 10, random_state = 42)),
              ('svc', make_pipeline(StandardScaler(), LinearSVC(random_state = 42)))]

# Meta Model stacked on top of base estimators
clf = StackingClassifier(estimators = estimators, final_estimator = LogisticRegression())

# Fit the model on traing data
stacking = clf.fit(X_train, y_train)

# Accuracy
stacking.score(X_test, y_test)

# Save the Stacking model 
pickle.dump(stacking, open('stacking_iris.pkl', 'wb'))


# Load the saved model
model = pickle.load(open('stacking_iris.pkl', 'rb'))
model


# Load test dataset for evaluation
test = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Ensemble_Techniques/Voting_and_Stacking/iris_test.csv")
test

pred = model.predict(test)
pred

