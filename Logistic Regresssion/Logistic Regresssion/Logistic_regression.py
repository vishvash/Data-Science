## Problem Statement
'''
The Insurance problem.

The insurance firm profits are defined by the number of claims made by the customers.

When an insurance company receives more claims, it usually means more payouts,
which can decrease profits. This can lead insurance firms to be more stringent 
or selective in approving claims, which might lead to disputes with customers.

This can lead to legal battles, which not only affect the company's financials
due to legal costs but can also impact its reputation.

This dynamic poses a significant challenge for insurance companies. 
They need to balance their risk management and profitability with customer 
satisfaction and legal compliance. Effective claim management, transparent 
policies, and good customer service can help in reducing the likelihood of 
disputes escalating to legal action.


**Objective(s):** Maximize the profits

**Constraints:** Maximize the customer satisfaction

**Success Criteria**

- **Business Success Criteria**: Improve the profits from anywh.ere between 10% to 20%.

- **ML Success Criteria**: Accuracy should be around 70% - 75%

- **Economic Success Criteria**: Reduce the legal complaints with appropriate 
risk management to increase the revenues by atleast 20%.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from feature_engine.outliers import Winsorizer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler #, MinMaxScaler

from sklearn.pipeline import Pipeline
import pickle, joblib

# import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.model_selection import train_test_split # train and test 
 
# import pylab as pl
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report

# SQL Integration
from sqlalchemy import create_engine, text

engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user = "root",    # user
                               pw = "1234",      # passwrd
                               db = "claims_db")) # database


# Load the offline data into Database to simulate client conditions
claims = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Logistic Regresssion/Logistic Regresssion/claimants.csv").convert_dtypes()
claims.info()
claims.to_sql('claims', con = engine, if_exists = 'replace', chunksize = 1000, index= False)


#### Read the Table (data) from MySQL database

sql = 'SELECT * FROM claims'

# Convert columns to best possible dtypes using
claimants = pd.read_sql_query(text(sql), engine.connect()).convert_dtypes()

claimants.head()

# Removing CASENUM
c1 = claimants.drop('CASENUM', axis = 1)
c1.info()
c1.describe()
c1.isna().sum()

# Predictors
X = c1[['CLMSEX', 'CLMINSUR', 'SEATBELT', 'CLMAGE', 'LOSS']]
X

# Target
y = c1[['ATTORNEY']]
y

# Segregating data based on their data types
numeric_features = X.select_dtypes(exclude = ['object']).columns
numeric_features

# Seperating Integer and Float data
numeric_features1 = X.select_dtypes(include = ['int64']).columns
numeric_features1

numeric_features2 = X.select_dtypes(include = ['float64']).columns
numeric_features2

# Imputation techniques to handle missing data
# Mode imputation for Integer (categorical) data
num_pipeline1 = Pipeline(steps=[('impute1', SimpleImputer(strategy = 'most_frequent'))])

# Mean imputation for Continuous (Float) data
num_pipeline2 = Pipeline(steps=[('impute2', SimpleImputer(strategy = 'mean'))])


# 1st Imputation Transformer
preprocessor = ColumnTransformer([
        ('mode', num_pipeline1, numeric_features1),
        ('mean', num_pipeline2, numeric_features2)])

print(preprocessor)

# Fit the data to train imputation pipeline model
impute_data = preprocessor.fit(X)

# Save the pipeline
joblib.dump(impute_data, 'impute')

# Transform the original data
X1 = pd.DataFrame(impute_data.transform(X), columns = X.columns).convert_dtypes()

X1.isna().sum()
X1.info()


# Multiple boxplots in a single visualization.
# Columns with larger scales affect other columns. 
# Below code ensures each column gets its own y-axis.

# pandas plot() function with parameters kind = 'box' and subplots = True

X1.plot(kind = 'box', subplots = True, sharey = False, figsize = (15, 8)) 
'''sharey True or 'all': x- or y-axis will be shared among all subplots.
False or 'none': each subplot x- or y-axis will be independent.'''
# increase spacing between subplots
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()

# CLMAGE and Loss features are continuous data with outliers 
# Ignore other categorical features

winsor = Winsorizer(capping_method = 'iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 1.5,
                          variables = ['CLMAGE', 'LOSS'])


outlier_pipeline = Pipeline(steps = [('winsor', winsor)])
outlier_pipeline


preprocessor1 = ColumnTransformer(transformers = [('wins', 
                                                   outlier_pipeline,
                                                   numeric_features)], 
                                  remainder = 'passthrough')
print(preprocessor1)


# Fit the data 
winz_data = preprocessor1.fit(X1)

# Save the pipeline
joblib.dump(winz_data, 'winzor')

X2 = pd.DataFrame(winz_data.transform(X1), columns = X1.columns).convert_dtypes()
X2.info()

# Boxplot
X2.plot(kind = 'box', subplots = True, sharey = False, figsize = (15, 8)) 
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()


# Address the scaling issue
scale_pipeline = Pipeline(steps=[('scale', StandardScaler())])
# scale_pipeline = Pipeline(steps=[('scale', MinMaxScaler())])

preprocessor2 = ColumnTransformer(transformers = [('num', 
                                                 scale_pipeline, numeric_features)], 
                                  remainder = 'passthrough')

print(preprocessor2)

scale = preprocessor2.fit(X2)

joblib.dump(scale, 'scale')

X3 = pd.DataFrame(scale.transform(X2), columns = X2.columns)
X3.columns
X3.info()
#######################


# Target variable
y.info()
# What is the difference between "Int64" and "int64"?
# One is a nullable integer dtype. The other is a numpy dtype.

y = y.astype('int')
y.info()


### Statsmodel 
# Building the model and fitting the data
logit_model = sm.Logit(y, X3).fit()

# Save the model
pickle.dump(logit_model, open('logistic.pkl', 'wb'))

# Summary
logit_model.summary()

logit_model.summary2() # for AIC


# Prediction
pred = logit_model.predict(X3)
pred  # Probabilities

# ROC Curve to identify the appropriate cutoff value
fpr, tpr, thresholds = roc_curve(y.ATTORNEY, pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold


auc = metrics.auc(fpr, tpr)
print("Area under the ROC curve : %f" % auc)

# Filling all the cells with zeroes
X3["pred"] = np.zeros(1340)

# taking threshold value and above the prob value will be treated as correct value 
X3.loc[pred > optimal_threshold, "pred"] = 1


# Confusion Matrix
confusion_matrix(X3.pred, y.ATTORNEY)

# Accuracy score of the model
print('Test accuracy = ', accuracy_score(X3.pred, y.ATTORNEY))

# Classification report
classification = classification_report(X3["pred"], y)
print(classification)

### PLOT FOR ROC
plt.plot(fpr, tpr, label = "AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc = 4)
plt.show()

################################################################

# Model evaluation - Data Split
x_train, x_test, y_train, y_test = train_test_split (X3.iloc[:, :5], y, 
                                                     test_size = 0.2, 
                                                     random_state = 0,
                                                     stratify = y)


# Fitting Logistic Regression to the training set  
logisticmodel = sm.Logit(y_train, x_train).fit()

# Evaluate on train data
y_pred_train = logisticmodel.predict(x_train)  
y_pred_train

# Metrics
# Filling all the cells with zeroes
y_train["pred"] = np.zeros(1072)

# taking threshold value and above the prob value will be treated as correct value 
y_train.loc[pred > optimal_threshold, "pred"] = 1

auc = metrics.roc_auc_score(y_train["ATTORNEY"], y_pred_train)
print("Area under the ROC curve : %f" % auc)

classification_train = classification_report(y_train["pred"], y_train["ATTORNEY"])
print(classification_train)

# confusion matrix 
confusion_matrix(y_train["pred"], y_train["ATTORNEY"])

# Accuracy score of the model
print('Train accuracy = ', accuracy_score(y_train["pred"], y_train["ATTORNEY"]))


# Validate on Test data
y_pred_test = logisticmodel.predict(x_test)  
y_pred_test

# Filling all the cells with zeroes
y_test["y_pred_test"] = np.zeros(268)

# Capturing the prediction binary values
y_test.loc[y_pred_test > optimal_threshold, "y_pred_test"] = 1

# classification report
classification1 = classification_report(y_test["y_pred_test"], y_test["ATTORNEY"])
print(classification1)

# confusion matrix 
confusion_matrix(y_test["y_pred_test"], y_test["ATTORNEY"])

# Accuracy score of the model
print('Test accuracy = ', accuracy_score(y_test["y_pred_test"], y_test["ATTORNEY"]))

#############################################

# Test the best model on new data
model1 = pickle.load(open('logistic.pkl', 'rb'))
impute = joblib.load('impute')
winzor = joblib.load('winzor')
minmax = joblib.load('scale')

# Load the new data
data = pd.read_excel(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Logistic Regresssion/Logistic Regresssion/claims_test.xlsx").convert_dtypes()

data = data.drop('CASENUM', axis = 1)
data.head()

# Engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}") #database
clean = pd.DataFrame(impute.transform(data), columns = data.columns).convert_dtypes()

clean1 = pd.DataFrame(winzor.transform(clean), columns = data.columns).convert_dtypes()

clean3 = pd.DataFrame(minmax.transform(clean1), columns = clean1.columns)

prediction = model1.predict(clean3)
prediction

# optimal_threshold=0.60
data["ATTORNEY"] = np.zeros(len(prediction))

# taking threshold value and above the prob value will be treated as correct value 
data.loc[prediction > optimal_threshold, "ATTORNEY"] = 1
data[['ATTORNEY']] = data[['ATTORNEY']].astype('int64')


