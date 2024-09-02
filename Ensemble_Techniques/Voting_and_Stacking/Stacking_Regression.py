'''
#CRISP-ML(Q) Framework: 

Business Understanding:

Business Problem: Significant proportion of people are unable to detect the 
presence of diabetes until it reaches adverse state because of lack of awareness. 

Business Objective: Maximize Early Detection of Diabetes
Business Constraint: Maximize Patient Convenience

Success Criteria:
Business: Maximize early diabetes detection by 50%
ML: Achieve an accuracy of more than 85%
Economic: Cost savings of more than $250K

Data Understanding: 
442 observations & 11 columns

10 columns are inputs & 1 column is output

1. Age
2. Sex
3. BMI
4. BP
Note: S1 to s6 are masked because of security reasons

6. Output - Target 

Result*: 	    Fasting Blood Sugar Test	Glucose Tolerance Test	    Random Blood Sugar Test
Diabetes:	    126 mg/dL or above	        200 mg/dL or above	        200 mg/dL or above
Prediabetes:	100 – 125 mg/dL	            140 – 199 mg/dL	            N/A
Normal:	        99 mg/dL or below	        140 mg/dL or below	        N/A


md/DL - milligrams per deciliter
'''


# Stacking Regression Using scikit-learn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
# from sklearn import metrics

# Load the dataset
diabetes = load_diabetes()

# # Load the dataframe
df_features = pd.DataFrame(data = diabetes.data, columns = diabetes.feature_names)

df_target = pd.DataFrame(data = diabetes.target, columns = ['target'])

final = pd.concat([df_features, df_target], axis = 1)
final


# ### Save the data frame into csv file
# final.to_csv('diabetes_new.csv', index = False)
# 
# final = pd.read_csv('diabetes_new.csv')
# 
# final
# final.info()

X = np.array(final.iloc[:, :10]) # Predictors
y = np.array(final['target']) # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)


# Base estimators
estimators = [("lr", RidgeCV()), ("svr", LinearSVR(random_state = 42))]

# Meta Model stacked on top of base estimators
reg = StackingRegressor(estimators = estimators,
                        final_estimator = RandomForestRegressor(n_estimators = 10,
                                                                random_state = 42))

stacking_reg = reg.fit(X_train, y_train)
stacking_reg

# Save the ML model
pickle.dump(stacking_reg, open('stacking_reg_diabetes.pkl', 'wb'))

# Load the saved model
model = pickle.load(open('stacking_reg_diabetes.pkl','rb'))


pred = model.predict(X_test)
pred

r2_score = model.score(X_test, y_test)

print(r2_score)

test = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Ensemble_Techniques/Voting_and_Stacking/diabetes_test.csv")

test_pred = model.predict(test)
test_pred
