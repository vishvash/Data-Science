'''CRISP-ML(Q)

a. Business & Data Understanding
    A construction firm wants to develop a suburban locality with new infrastructure but they might incur losses if they cannot sell the properties. To overcome this, they consult an analytics firm to get insights into how densely the area is populated and the income levels of residents. Use the Support Vector Machines algorithm on the given dataset and draw out insights and also comment on the viability of investing in that area.

    i. Business Objective - Maximize viablity of investing
    ii. Business Constraint - Minimize Prediction Errors

    Success Criteria:
    1. Business Success Criteria - Increase finding viable areas by atleast 20%
    2. ML Success Criteria - Achieve a prediction accuracy of atleast by 80%
    3. Economic Success Criteria - Increase the profit atleast by 15%

    Data Collection - Data is collected to find insights into how densely the area is populated and the income levels of residents

    Metadata Description:
    Feature Name       Description                                  Type          Relevance
    --------------     -----------------------------------------     ------------- --------------
    age               Age of the individual                        Quantitative  Relevant
    workclass         Type of work class                           Nominal       Relevant
    education         Level of education                           Nominal       Relevant
    educationno       Number of years of education                 Quantitative  Relevant
    maritalstatus     Marital status                               Nominal       Relevant
    occupation        Occupation                                   Nominal       Relevant
    relationship      Relationship status                          Nominal       Relevant
    race              Race of the individual                       Nominal       Relevant
    sex               Gender                                       Nominal       Relevant
    capitalgain       Capital gain                                 Quantitative  Irrelevant
    capitalloss       Capital loss                                 Quantitative  Irrelevant
    hoursperweek      Number of hours worked per week              Quantitative  Relevant
    native            Native country                               Nominal       Relevant
    Salary            Salary level                                 Nominal       Relevant

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
import pickle, joblib

from sqlalchemy import create_engine, text


user = 'root'  # user name
pw = '1234'  # password
db = 'salary_db'  # database name
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")


salary = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/Data Science/SVM_classifier/Assignment/SVM/SalaryData_Train.csv")

salary.to_sql('salary_svm', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

sql = 'select * from salary_svm;'
salary = pd.read_sql_query(text(sql), engine.connect()).head(5000)

salary.describe()

salary.drop(columns = ["capitalgain", "capitalloss"], inplace = True)


salary['High_Sal'] = np.where(salary.Salary == ' >50K', 1, 0)

salary.drop(columns = ["Salary"], inplace = True)

# Predictors and Target
X = salary.iloc[:, :11]
Y = salary.iloc[:, 11]


# Define numeric and categorical features
numeric_features = X.select_dtypes(exclude=['object']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Outlier Treatment

# Multiple boxplots in a single visualization.
# Columns with larger scales affect other columns. 
# Below code ensures each column gets its own y-axis.
X[numeric_features].plot(kind = 'box', subplots = True, sharey = False, figsize = (15, 8)) 
'''sharey True or 'all': x- or y-axis will be shared among all subplots.
False or 'none': each subplot x- or y-axis will be independent.'''
# increase spacing between subplots
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()

X.info()



# Pipeline for numerical feature preprocessing
numeric_pipeline = Pipeline(steps=[
    ('winsor', Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables= list(numeric_features))),
    ('impute', SimpleImputer(strategy='mean')),
    ('scale', MinMaxScaler())
    
])

# Pipeline for categorical feature preprocessing
categorical_pipeline = Pipeline(steps=[
    ('onehot', OneHotEncoder())
])

# Preprocessor using ColumnTransformer
preprocessor = ColumnTransformer([('numeric', numeric_pipeline, numeric_features),
    ('categorical', categorical_pipeline, categorical_features)],  remainder='passthrough')

# Fit and transform the preprocessor to the data
preprocessed = preprocessor.fit(X)


# Save the data preprocessing pipeline
joblib.dump(preprocessed, 'preprocessor.pkl')

clean_data1 = pd.DataFrame(preprocessed.transform(X).toarray(), columns = list(preprocessed.get_feature_names_out()))

clean_data1.info()

# Boxplot
clean_data1[["numeric__age", "numeric__educationno", "numeric__hoursperweek"]].plot(kind = 'box', subplots = True, sharey = False, figsize = (15, 8)) 
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()

X2 = clean_data1

# Data Partition into Train and Test
train_X, test_X, train_y, test_y = train_test_split(X2, Y, test_size = 0.2, stratify = Y)


# Support Vector Classifier
# SVC with linear kernel trick
model_linear = SVC(kernel = "linear")
model1 = model_linear.fit(train_X, train_y)

a = model_linear.coef_

# print(model_linear.decision_function_shape)

pred_test_linear = model_linear.predict(test_X)

# Accuracy
np.mean(pred_test_linear == test_y)


### Hyperparameter Optimization
# RandomizedSearchCV

# Base model
model = SVC()

# Parameters set
parameters = {'C': [0.1, 1, 10, 100], 
              'gamma': [1, 0.1, 0.01, 0.001],
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}

# Randomized Search Technique for exhaustive search for best model
rand_search =  RandomizedSearchCV(model, parameters, n_iter = 10, 
                                  n_jobs = 3, cv = 3, scoring = 'accuracy', random_state = 0)
  
# Fitting the model for grid search
randomised = rand_search.fit(train_X, train_y)

# Best parameters
randomised.best_params_

# Best Model
best = randomised.best_estimator_

# Evaluate on Test data
pred_test = best.predict(test_X)

np.mean(pred_test == test_y)


# Saving the best model - rbf kernel model 
pickle.dump(best, open('svc_rcv.pkl', 'wb'))


##########  New Data Prediction ####

data = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/Data Science/SVM_classifier/Assignment/SVM/test.csv")

data.drop(columns = ["capitalgain", "capitalloss"], inplace = True)

numeric_features = data.select_dtypes(exclude = ['object']).columns
  
model1 = pickle.load(open('svc_rcv.pkl', 'rb'))
preprocessed = joblib.load('preprocessor.pkl')

clean = pd.DataFrame(preprocessed.transform(data).toarray(), columns = list(preprocessed.get_feature_names_out()))

prediction = pd.DataFrame(model1.predict(clean), columns = ['sal_pred'])

final = pd.concat([prediction, data], axis = 1)
