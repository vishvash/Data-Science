'''
    1.a. Business problem: A cloth manufacturing company is interested to know about the different attributes contributing to high sales. 
        i. Business Objectives: Maximize the sales
        ii. Business Constraints: Minimize the advertising
        Success Criteria:
        i. Business success criteria: Increase the efficient of marketing by 10%
        ii. ML success criteria: Achieve an accuracy of at least 70%
        iii. Economic success criteria: Increase the sales by at least by 20%
    1.b. Data Collection: Bank -> 400 sales data, 11 variables (10 Inputs and 1 Ouput)
    2. Data Preprocessing - Cleansing & EDA / Descriptive Analytics
    3. Model Building - Experiment with different models alongside Hyperparameter tuning
    4. Evaluation - Not just model evaluation based on accuracy but we also need 
       to evaluate business & economic success criteria
    5. Model Deployment (Flask)
    6. Monitoring & Maintenance (Prediction results to the database - MySQL / MS SQL)
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from feature_engine.outliers import Winsorizer

from sklearn.model_selection import train_test_split

import joblib
import pickle

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn import datasets, linear_model, neighbors, ensemble #, svm, naive_bayes
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

# pip install xgboost
import xgboost as xgb

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate


# Hyperparameter optimization
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

import sklearn.metrics as skmet


data = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Ensemble_Techniques/Assignment/Ensemble Models/ClothCompany_Data (1).csv")

from sqlalchemy import create_engine, text
from urllib.parse import quote

# creating engine to connect MySQL database
user = 'root' # user name
pw = quote('1234') # password
db = 'sales_db' # database

engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

data.to_sql('sales', con = engine, if_exists = 'replace', chunksize = 1000, index = False )


sql = "SELECT * FROM sales;"

# for sqlalchmey 1.4.x version
# df = pd.read_sql_query(sql, engine)

# for sqlalchmey 2.x version 
df = pd.read_sql_query(text(sql), engine.connect())


# df = pd.read_csv(r"movies_classification.csv")
df.head()

df.info()


# Define the numerical ranges and corresponding labels for the categories
sales_bins = [0, 5, 10, float('inf')]
sales_labels = [ 'Low', 'Medium', 'High']

# Convert numerical "Sales" df to categorical using cut function
df['Sales_Category'] = pd.cut(df['Sales'], bins=sales_bins, labels=sales_labels, right=False)

# Drop the original "Sales" column if needed
df.drop(columns=['Sales'], inplace=True)



# AutoEDA
# D-Tale
#########
# pip install dtale
import dtale

d = dtale.show(df)
d.open_browser()

# or
# AutoEDA
# Sweetviz
import sweetviz
my_report = sweetviz.analyze(df)

my_report.show_html('Report1.html')
#########

df['Sales_Category'].unique()

df['Sales_Category'].value_counts()


# Data split into Input and Output
X = df.iloc[:, :10] # Predictors 

y = df['Sales_Category'] # Target 

df.info()

# #### Separating Numeric and Non-Numeric columns
numeric_features = X.select_dtypes(exclude = ['object']).columns
numeric_features

categorical_features = X.select_dtypes(include=['object']).columns
categorical_features


# ### Data Preprocessing

# Numeric_features
# ### Imputation to handle missing values 
# ### MinMaxScaler to convert the magnitude of the columns to a range of 0 to 1
num_pipeline = Pipeline(steps = [('impute', SimpleImputer(strategy = 'mean')), ('scale', MinMaxScaler())])


# ### Encoding - One Hot Encoder to convert Categorical data to Numeric values
# Categorical features
encoding_pipeline = Pipeline([('onehot', OneHotEncoder(drop = 'first'))])

# Creating a transformation of variable with ColumnTransformer()
preprocessor = ColumnTransformer(transformers = [('num', num_pipeline, numeric_features), ('categorical', encoding_pipeline, categorical_features)])

imp_enc_scale = preprocessor.fit(X)

# #### Save the pipeline model using joblib
joblib.dump(imp_enc_scale, 'imp_enc_scale')

import os
os.getcwd()

cleandata = pd.DataFrame(imp_enc_scale.transform(X), columns = imp_enc_scale.get_feature_names_out())
cleandata

# Note: If you get any error then update the scikit-learn library version & restart the kernel to fix it

# ### Outlier Analysis

# Multiple boxplots in a single visualization.
# Columns with larger scales affect other columns. 
# Below code ensures each column gets its own y-axis.

# pandas plot() function with parameters kind = 'box' and subplots = True

cleandata.iloc[:, 0:7].plot(kind = 'box', subplots = True, sharey = False, figsize = (10, 6)) 
'''sharey True or 'all': x- or y-axis will be shared among all subplots.
False or 'none': each subplot x- or y-axis will be independent.'''
# Increase spacing between subplots
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()


cleandata.iloc[:, 0:7].columns

# #### Outlier analysis: Columns 'months_loan_duration', 'amount', and 'age' are continuous, hence outliers are treated
winsor = Winsorizer(capping_method = 'iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 1.5,
                          variables = list(cleandata.iloc[:, 0:7].columns))

outlier = winsor.fit(cleandata.iloc[:, 0:7])

# Save the winsorizer model 
joblib.dump(outlier, 'winsor')

cleandata.iloc[:, 0:7] = outlier.transform(cleandata.iloc[:, 0:7])

# Clean data
cleandata


# Verify for outliers
cleandata.iloc[:, 0:7].plot(kind = 'box', subplots = True, sharey = False, figsize = (10, 6)) 
# increase spacing between subplots
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()

clean_data = cleandata
target = y

# Splitting data into training and testing data set
X_train, X_test, Y_train, Y_test = train_test_split(clean_data, target, test_size = 0.2, 
                                                    stratify = target, random_state = 0) 
### Decision Tree Model
model = DT(criterion = 'entropy')
model.fit(X_train, Y_train)


# Prediction on Test Data
preds = model.predict(X_test)

preds


# Accuracy
print(accuracy_score(Y_test, preds))

pd.crosstab(Y_test, preds, rownames = ['Actual'], colnames = ['Predictions']) 


### Hyperparameter Optimization
# create a dictionary of all hyperparameters to be experimented
param_grid = { 'criterion':['gini', 'entropy'], 'max_depth': np.arange(3, 15)}

# Decision tree model
dtree_model = DT()

# GridsearchCV with cross-validation to perform experiments with parameters set
dtree_gscv = GridSearchCV(dtree_model, param_grid, cv = 5, scoring = 'accuracy',
                          return_train_score = False, verbose = 1)


# Train the model with Grid search optimization technique
dtree_gscv.fit(X_train, Y_train)

# The best set of parameter values
dtree_gscv.best_params_


# Model with best parameter values
DT_best = dtree_gscv.best_estimator_
DT_best

# Prediction on Test Data
preds1 = DT_best.predict(X_test)
preds1

# Model evaluation
# Cross Table (Confusion Matrix)
pd.crosstab(Y_test, preds, rownames = ['Actual'], colnames= ['Predictions']) 

# Accuracy
print(accuracy_score(Y_test, preds))


# ### Save the Best Model with pickel library
pickle.dump(DT_best, open('DT.pkl', 'wb'))


# Base Model 1
### k-Nearest Neighbors (k-NN) with GridSearchCV
knn = neighbors.KNeighborsClassifier()

params_knn = {'n_neighbors': np.arange(1, 25)}

knn_gs = GridSearchCV(knn, params_knn, cv = 5)

knn_gs.fit(X_train, Y_train)
knn_gs.best_params_

knn_best = knn_gs.best_estimator_



# Base Model 2
### Random Forest Classifier with GridSearchCV
rf = ensemble.RandomForestClassifier(random_state = 0)

params_rf = {'n_estimators': [50, 100, 200]}

rf_gs = GridSearchCV(rf, params_rf, cv = 5)

rf_gs.fit(X_train, Y_train)
rf_gs.best_params_

rf_best = rf_gs.best_estimator_



# Base Model 3
### Logistic Regression with GridSearchCV
log_reg = linear_model.LogisticRegression()

C = np.logspace(1, 4, 10)
params_lr = dict(C = C)

lr_gs = GridSearchCV(log_reg, params_lr, cv = 5)

lr_gs.fit(X_train, Y_train)
lr_gs.best_estimator_

lr_best = lr_gs.best_estimator_



# Combine all three Based models
estimators = [('knn', knn_best), ('rf', rf_best), ('log_reg', lr_best)]



# Hard/Majority Voting
# # VotingClassifier with voting = "hard" parameter
ensemble_H = VotingClassifier(estimators, voting = "hard")

# Fit classifier with the training data
hard_voting = ensemble_H.fit(X_train, Y_train)

# Save the voting classifier
pickle.dump(hard_voting, open('hard_voting.pkl', 'wb'))

# Loading a saved model
model = pickle.load(open('hard_voting.pkl', 'rb'))

print("knn_gs.score: ", knn_best.score(X_test, Y_test))
# Output: knn_gs.score:
    
print("rf_gs.score: ", rf_best.score(X_test, Y_test))
# Output: rf_gs.score:

print("log_reg.score: ", lr_best.score(X_test, Y_test))
# Output: log_reg.score:

# Hard Voting Ensembler
print("Hard Voting Ensemble Score: ", ensemble_H.score(X_test, Y_test))
# Output: ensemble.score:



#############################################################
# Soft Voting
# VotingClassifier with voting = "soft" parameter
ensemble_S = VotingClassifier(estimators, voting = "soft")

soft_voting = ensemble_S.fit(X_train, Y_train)

# Save model
pickle.dump(soft_voting, open('soft_voting.pkl', 'wb'))


# Load the saved model
model = pickle.load(open('soft_voting.pkl', 'rb'))

print("knn_gs.score: ", knn_gs.score(X_test, Y_test))
# Output: knn_gs.score:

print("rf_gs.score: ", rf_gs.score(X_test, Y_test))
# Output: rf_gs.score:

print("log_reg.score: ", lr_gs.score(X_test, Y_test))
# Output: log_reg.score:

print("Soft Voting Ensemble Score: ", ensemble_S.score(X_test, Y_test))
# Output: ensemble.score:

# Base estimators
estimators = [('rf', RandomForestClassifier(n_estimators = 10, random_state = 42)),
              ('svc', LinearSVC(random_state = 42))]


# Meta Model stacked on top of base estimators
clf = StackingClassifier(estimators = estimators, final_estimator = LogisticRegression())

# Fit the model on traing data
stacking = clf.fit(X_train, Y_train)

# Accuracy
stacking.score(X_test, Y_test)

# Save the Stacking model 
pickle.dump(stacking, open('stacking_cloth.pkl', 'wb'))


# Load the saved model
model = pickle.load(open('stacking_cloth.pkl', 'rb'))
model

# # Bagging Classifier Model
# from sklearn.ensemble import BaggingClassifier
# decision tree defined first
# from sklearn import tree
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score

clftree = tree.DecisionTreeClassifier()


bag_clf = BaggingClassifier(estimator = clftree, n_estimators = 500,
                            bootstrap = True, n_jobs = -1, random_state = 42)

# Fit the model 
bagging = bag_clf.fit(X_train, Y_train)

print(confusion_matrix(Y_train, bagging.predict(X_train)))
print(accuracy_score(Y_train, bagging.predict(X_train)))
print('\n')
print(confusion_matrix(Y_test, bagging.predict(X_test)))
print(accuracy_score(Y_test, bagging.predict(X_test)))

# Saving the best model
pickle.dump(bagging, open('baggingmodel.pkl', 'wb'))


# ## Cross Validation implementation
# from sklearn.model_selection import cross_validate

def cross_validation(model, _X, _y, _cv=5):
    _scoring = ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(estimator=model,
                           X=_X,
                           y=_y,
                           cv=_cv,
                           scoring=_scoring,
                           return_train_score=True)

    return pd.DataFrame({"Training Accuracy scores": results['train_accuracy'],
          "Mean Training Accuracy": results['train_accuracy'].mean()*100,
          "Training Precision scores": results['train_precision'],
          "Mean Training Precision": results['train_precision'].mean(),
          "Training Recall scores": results['train_recall'],
          "Mean Training Recall": results['train_recall'].mean(),
          "Training F1 scores": results['train_f1'],
          "Mean Training F1 Score": results['train_f1'].mean(),
          "Validation Accuracy scores": results['test_accuracy'],
          "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
          "Validation Precision scores": results['test_precision'],
          "Mean Validation Precision": results['test_precision'].mean(),
          "Validation Recall scores": results['test_recall'],
          "Mean Validation Recall": results['test_recall'].mean(),
          "Validation F1 scores": results['test_f1'],
          "Mean Validation F1 Score": results['test_f1'].mean()
          })

# Call the above custom function
Bagging_cv_scores = cross_validation(bag_clf, X_train, Y_train, 5)
Bagging_cv_scores


# Visualization custom function
def plot_result(x_label, y_label, plot_title, train_data, val_data):
        # Set size of plot
        plt.figure(figsize=(12,6))
        labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold"]
        X_axis = np.arange(len(labels))
        # ax = plt.gca()
        plt.ylim(0.40000, 1)
        plt.bar(X_axis-0.2, train_data, 0.4, color='blue', label='Training')
        plt.bar(X_axis+0.2, val_data, 0.4, color='red', label='Validation')
        plt.title(plot_title, fontsize=30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()

# get_ipython().run_line_magic('matplotlib', 'inline')
model_name = "Bagging Classifier"
plot_result(model_name,
            "Accuracy",
            "Accuracy scores in 5 Folds",
            Bagging_cv_scores["Training Accuracy scores"],
            Bagging_cv_scores["Validation Accuracy scores"])


# ## Random Forest Model
# from sklearn.ensemble import RandomForestClassifier

rf_Model = RandomForestClassifier()

# #### Hyperparameters
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 80, num = 10)]

# Number of features to consider at every split
max_features = ['log2', 'sqrt']

# Maximum number of levels in tree
max_depth = [2, 4]

# Minimum number of samples required to split a node
min_samples_split = [2, 5]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2]

# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the param grid
param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

print(param_grid)


# ### Hyperparameter optimization with GridSearchCV
rf_Grid = GridSearchCV(estimator = rf_Model, param_grid = param_grid, cv = 10, verbose = 1, n_jobs = -1)

rf_Grid.fit(X_train, Y_train)

rf_Grid.best_params_

cv_rf_grid = rf_Grid.best_estimator_


# ## Check Accuracy
# Evaluation on Test Data
test_pred = cv_rf_grid.predict(X_test)

accuracY_test = np.mean(test_pred == Y_test)
accuracY_test

cm = skmet.confusion_matrix(Y_test, test_pred)

cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['Low', 'Medium', 'High'])
cmplot.plot()
cmplot.ax_.set(title = 'Sales Detection Confusion Matrix', 
               xlabel = 'Predicted Value', ylabel = 'Actual Value')


print (f'Train Accuracy - : {rf_Grid.score(X_train, Y_train):.3f}')
print (f'Test Accuracy - : {rf_Grid.score(X_test, Y_test):.3f}')


# RandomizedSearchCV
# ### Hyperparameter optimization with RandomizedSearchCV
rf_Random = RandomizedSearchCV(estimator = rf_Model, param_distributions = param_grid, cv = 10, verbose = 2, n_jobs = -1)

rf_Random.fit(X_train, Y_train)

rf_Random.best_params_

cv_rf_random = rf_Random.best_estimator_

# Evaluation on Test Data
test_pred_random = cv_rf_random.predict(X_test)

accuracY_test_random = np.mean(test_pred_random == Y_test)
accuracY_test_random


print (f'Train Accuracy - : {rf_Random.score(X_train, Y_train):.3f}')
print (f'Test Accuracy - : {rf_Random.score(X_test, Y_test):.3f}')


# ## Save the best model from Randomsearch CV approach
pickle.dump(cv_rf_random, open('rfc.pkl', 'wb'))


# ## Cross Validation implementation
# from sklearn.model_selection import cross_validate
def cross_validation(model, _X, _y, _cv = 5):
    _scoring = ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(estimator=model,
                           X = _X,
                           y = _y,
                           cv = _cv,
                           scoring = _scoring,
                           return_train_score = True)

    return pd.DataFrame({"Training Accuracy scores": results['train_accuracy'],
          "Mean Training Accuracy": results['train_accuracy'].mean()*100,
          "Training Precision scores": results['train_precision'],
          "Mean Training Precision": results['train_precision'].mean(),
          "Training Recall scores": results['train_recall'],
          "Mean Training Recall": results['train_recall'].mean(),
          "Training F1 scores": results['train_f1'],
          "Mean Training F1 Score": results['train_f1'].mean(),
          "Validation Accuracy scores": results['test_accuracy'],
          "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
          "Validation Precision scores": results['test_precision'],
          "Mean Validation Precision": results['test_precision'].mean(),
          "Validation Recall scores": results['test_recall'],
          "Mean Validation Recall": results['test_recall'].mean(),
          "Validation F1 scores": results['test_f1'],
          "Mean Validation F1 Score": results['test_f1'].mean()
          })


Random_forest_result = cross_validation(cv_rf_random, X_train, Y_train, 5)
Random_forest_result


def plot_result(x_label, y_label, plot_title, train_data, val_data):
        # Set size of plot
        plt.figure(figsize=(12, 6))
        labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold"]
        X_axis = np.arange(len(labels))
        plt.ylim(0.40000, 1)
        plt.bar(X_axis - 0.2, train_data, 0.1, color = 'blue', label = 'Training')
        plt.bar(X_axis + 0.2, val_data, 0.1, color = 'red', label = 'Validation')
        plt.title(plot_title, fontsize=30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()


model_name = "RandomForestClassifier"
plot_result(model_name,
            "Accuracy",
            "Accuracy scores in 5 Folds",
            Random_forest_result["Training Accuracy scores"],
            Random_forest_result["Validation Accuracy scores"])


# #  AdaBoosting
# from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(learning_rate = 0.02, n_estimators = 5000)

ada_clf1 = ada_clf.fit(X_train, Y_train)

predictions = ada_clf1.predict(X_test)

# Evaluation on Testing Data
confusion_matrix(Y_test, predictions)

accuracy_score(Y_test, predictions)

# Evaluation on Training Data
accuracy_score(Y_train, ada_clf1.predict(X_train))

# Saving the best model
pickle.dump(ada_clf1, open('adaboost.pkl','wb'))


## GradientBoosting
# from sklearn.ensemble import GradientBoostingClassifier
boost_clf = GradientBoostingClassifier()
boost_clf1 = boost_clf.fit(X_train, Y_train)

grad_pred = boost_clf1.predict(X_test)

print(confusion_matrix(Y_test, grad_pred))
print(accuracy_score(Y_test, grad_pred))

print(confusion_matrix(Y_train, boost_clf1.predict(X_train)))
print(accuracy_score(Y_train, boost_clf1.predict(X_train)))


# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier

# Hyperparameters
boost_clf2 = GradientBoostingClassifier(learning_rate = 0.02, n_estimators = 1000, max_depth = 1)

boost_clf_p = boost_clf2.fit(X_train, Y_train)

grad_pred_p = boost_clf_p.predict(X_test)

# Evaluation on Testing Data
print(confusion_matrix(Y_test, grad_pred_p))
print('\n')
print(accuracy_score(Y_test,grad_pred_p))


# Evaluation on Training Data
print(confusion_matrix(Y_train, boost_clf_p.predict(X_train)))
accuracy_score(Y_train, boost_clf_p.predict(X_train))

# Save the ML model
pickle.dump(boost_clf_p, open('gradiantboostparam.pkl', 'wb'))

grad_model_p = pickle.load(open('gradiantboostparam.pkl', 'rb'))


## XGBoosting
# pip install xgboost
# import xgboost as xgb
xgb_clf = xgb.XGBClassifier(max_depth = 5, n_estimators = 10000, 
                            learning_rate = 0.3, n_jobs = -1)

# n_jobs – Number of parallel threads used to run xgboost.
# learning_rate (float) – Boosting learning rate (xgb’s “eta”)

from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode the string classes into numerical labels
Y_train_encoded = label_encoder.fit_transform(Y_train)

Y_test_encoded = label_encoder.fit_transform(Y_test)

xgb_clf1 = xgb_clf.fit(X_train, Y_train_encoded)

xgb_pred = xgb_clf1.predict(X_test)

# Evaluation on Testing Data
print(confusion_matrix(Y_test_encoded, xgb_pred))

accuracy_score(Y_test_encoded, xgb_pred)

# Feature Importance Plot
# "F score", also known as the "feature importance score".
# F-score of a feature is calculated based on how often the feature is used to 
# split the data across all trees in the model. 
# In simple terms, it reflects how useful or relevant a feature is for the 
# model's decision-making process.
xgb.plot_importance(xgb_clf)

fi = pd.DataFrame(xgb_clf1.feature_importances_.reshape(1, -1), columns = X_train.columns)
fi

# Save the ML model
pickle.dump(xgb_clf1, open('xgb.pkl', 'wb'))

xgb_model = pickle.load(open('xgb.pkl', 'rb'))


## RandomizedSearchCV for XGB
xgb_clf = xgb.XGBClassifier(n_estimators = 500, learning_rate = 0.1, random_state = 42)

# Grid Search
param_test1 = {'max_depth': range(3,10,2), 'gamma': [0.1, 0.2, 0.3],
               'subsample': [0.8, 0.9], 'colsample_bytree': [0.8, 0.9],}


xgb_RandomGrid = RandomizedSearchCV(estimator = xgb_clf, 
                                    param_distributions = param_test1, 
                                    cv = 5, verbose = 2, n_jobs = -1)

Randomized_search1 = xgb_RandomGrid.fit(X_train, Y_train_encoded)

cv_xg_clf = Randomized_search1.best_estimator_
cv_xg_clf

randomized_pred = Randomized_search1.predict(X_test)


# Evaluation on Testing Data with model with hyperparameter
accuracy_score(Y_test_encoded, randomized_pred)

Randomized_search1.best_params_

randomized_pred_1 = Randomized_search1.predict(X_train)


# Evaluation on Training Data with model with hyperparameters
accuracy_score(Y_train_encoded, randomized_pred_1)

pickle.dump(cv_xg_clf, open('Randomizedsearch_xgb.pkl', 'wb'))
