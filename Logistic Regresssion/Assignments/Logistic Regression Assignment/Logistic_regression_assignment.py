## Problem Statement
'''
In this time and age of widespread internet usage, effective and targeted marketing plays a vital role. A marketing company would like to develop a strategy by analyzing its customer data. For this, data like age, location, time of activity, etc. have been collected to determine whether a user will click on an ad or not. Perform Logistic Regression on the given data to predict whether a user will click on an ad or not. 


**Objective(s):** Maximize the clickrate

**Constraints:** Maximize the ad relevancy

**Success Criteria**

- **Business Success Criteria**: Improve the clickrates anywhere between 10% to 20%.

- **ML Success Criteria**: Accuracy should be around 70% - 75%

- **Economic Success Criteria**: Increase the ad revenues by atleast 20%.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from feature_engine.outliers import Winsorizer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler #, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder


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
                               db = "ctr_db")) # database


# Load the offline data into Database to simulate client conditions
clicks = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Logistic Regresssion/Assignments/Logistic Regression Assignment/advertising.csv").convert_dtypes()
clicks.info()
clicks.to_sql('clicks', con = engine, if_exists = 'replace', chunksize = 1000, index= False)


#### Read the Table (data) from MySQL database

sql = 'SELECT * FROM clicks'

# Convert columns to best possible dtypes using
adclicks = pd.read_sql_query(text(sql), engine.connect()).convert_dtypes()

adclicks.head()

adclicks.Ad_Topic_Line.duplicated().sum()
adclicks.City.duplicated().sum()
adclicks.Country.duplicated().sum()

c1 = adclicks.drop('Ad_Topic_Line', axis = 1)
c1 = c1.drop('City', axis = 1)
c1 = c1.drop('Country', axis = 1)

# Convert 'Timestamp' column to datetime format
c1['Timestamp'] = pd.to_datetime(c1['Timestamp'])

# Convert the datetime values to Unix time (seconds since epoch)
c1['Unix_Time'] = (c1['Timestamp'] - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)

# Now, 'df' contains a single numerical column 'Unix_Time' representing the timestamp

c1 = c1.drop('Timestamp', axis = 1)

c1.info()
c1.describe()
c1.isna().sum()
c1.head()

# Predictors
X = c1.iloc[:, [*range(0, 5), 6]]
X

# Target
# y = c1[['Clicked_on_Ad']]
y = c1.iloc[:, [5]]
y

# # Convert columns to 'object' data type
# X['City'] = X['City'].astype('object')
# X['Country'] = X['Country'].astype('object')

X.info()

# Segregating data based on their data types
numeric_features = X.select_dtypes(exclude = ['object', 'string']).columns
# numeric_features = X.select_dtypes(exclude = ['object']).columns
numeric_features

# Seperating Integer and Float data
numeric_features1 = X.select_dtypes(include = ['int64']).columns
numeric_features1

numeric_features2 = X.select_dtypes(include = ['float64']).columns
numeric_features2

# categ_features = X.select_dtypes(include = ['object', 'string']).columns
# categ_features

# Imputation techniques to handle missing data
# Mode imputation for Integer (categorical) data
num_pipeline1 = Pipeline(steps=[('impute1', SimpleImputer(strategy = 'most_frequent'))])

# Mean imputation for Continuous (Float) data
num_pipeline2 = Pipeline(steps=[('impute2', SimpleImputer(strategy = 'mean'))])

# One-hot encoding for categorical features
# cat_pipeline = Pipeline(steps=[('onehot', OneHotEncoder(drop = 'first'))])


# 1st Imputation Transformer
preprocessor = ColumnTransformer([
        ('mode', num_pipeline1, numeric_features1),
        ('mean', num_pipeline2, numeric_features2)])
        # ('onehot', cat_pipeline, categ_features)])

print(preprocessor)

# Fit the data to train imputation pipeline model
impute_data = preprocessor.fit(X)

# Save the pipeline
joblib.dump(impute_data, 'impute')


# Transform the original data
X1 = pd.DataFrame(impute_data.transform(X), columns = X.columns).convert_dtypes()


# traff = pd.DataFrame(processed3.transform(df1).toarray(), columns = list(processed3.get_feature_names_out()))



X1.isna().sum()
X1.info()


# Multiple boxplots in a single visualization.
# Columns with larger scales affect other columns. 
# Below code ensures each column gets its own y-axis.

# pandas plot() function with parameters kind = 'box' and subplots = True

X1.iloc[:,0:6].plot(kind = 'box', subplots = True, sharey = False, figsize = (15, 8)) 
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
                          variables = ['Area_Income'])


outlier_pipeline = Pipeline(steps = [('winsor', winsor)])
outlier_pipeline


preprocessor1 = ColumnTransformer(transformers = [('wins', 
                                                   outlier_pipeline,
                                                   ['Area_Income'])], 
                                  remainder = 'passthrough')
print(preprocessor1)

# print(X1.iloc[:,0:10].columns)

# Fit the data 
winz_data = preprocessor1.fit(X1)

# Save the pipeline
joblib.dump(winz_data, 'winzor')

X2 = pd.DataFrame(winz_data.transform(X1), columns = X1.columns).convert_dtypes()
X2.info()

# Boxplot
X2.iloc[:,0:6].plot(kind = 'box', subplots = True, sharey = False, figsize = (15, 8)) 
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()


# Address the scaling issue
scale_pipeline = Pipeline(steps=[('scale', StandardScaler())])
# scale_pipeline = Pipeline(steps=[('scale', MinMaxScaler())])

# print(X1.iloc[:,0:10].columns)

scalable_features = X2.iloc[:,0:6].columns


preprocessor2 = ColumnTransformer(transformers = [('num', 
                                                 scale_pipeline, scalable_features)], 
                                  remainder = 'passthrough')

print(preprocessor2)

scale = preprocessor2.fit(X2)

joblib.dump(scale, 'scale')

X3 = pd.DataFrame(scale.transform(X2), columns = X2.columns)
X3.columns
X3.info()

# Convert X3 to float dtype
X3 = X3.astype(float)

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
fpr, tpr, thresholds = roc_curve(y.Clicked_on_Ad, pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold


auc = metrics.auc(fpr, tpr)
print("Area under the ROC curve : %f" % auc)

# Filling all the cells with zeroes
X3["pred"] = np.zeros(1000)

# taking threshold value and above the prob value will be treated as correct value 
X3.loc[pred > optimal_threshold, "pred"] = 1


# Confusion Matrix
confusion_matrix(X3.pred, y.Clicked_on_Ad)

# Accuracy score of the model
print('Test accuracy = ', accuracy_score(X3.pred, y.Clicked_on_Ad))

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
x_train, x_test, y_train, y_test = train_test_split (X3.iloc[:, :6], y, 
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
y_train["pred"] = np.zeros(800)

# taking threshold value and above the prob value will be treated as correct value 
y_train.loc[pred > optimal_threshold, "pred"] = 1

auc = metrics.roc_auc_score(y_train["Clicked_on_Ad"], y_pred_train)
print("Area under the ROC curve for train data : %f" % auc)

classification_train = classification_report(y_train["pred"], y_train["Clicked_on_Ad"])
print(classification_train)

# confusion matrix 
confusion_matrix(y_train["pred"], y_train["Clicked_on_Ad"])

# Accuracy score of the model
print('Train accuracy = ', accuracy_score(y_train["pred"], y_train["Clicked_on_Ad"]))


# Validate on Test data
y_pred_test = logisticmodel.predict(x_test)  
y_pred_test

# Filling all the cells with zeroes
y_test["y_pred_test"] = np.zeros(200)

# Capturing the prediction binary values
y_test.loc[y_pred_test > optimal_threshold, "y_pred_test"] = 1

# classification report
classification1 = classification_report(y_test["y_pred_test"], y_test["Clicked_on_Ad"])
print(classification1)

# confusion matrix 
confusion_matrix(y_test["y_pred_test"], y_test["Clicked_on_Ad"])

# Accuracy score of the model
print('Test accuracy = ', accuracy_score(y_test["y_pred_test"], y_test["Clicked_on_Ad"]))

#############################################

# Test the best model on new data
model1 = pickle.load(open('logistic.pkl', 'rb'))
impute = joblib.load('impute')
winzor = joblib.load('winzor')
minmax = joblib.load('scale')

# Load the new data
data = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Logistic Regresssion/Assignments/Logistic Regression Assignment/advertising_test.csv").convert_dtypes()


data = data.drop(['Ad_Topic_Line', 'City', 'Country'], axis = 1)

# Convert 'Timestamp' column to datetime format
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Convert the datetime values to Unix time (seconds since epoch)
data['Unix_Time'] = (data['Timestamp'] - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)

# Now, 'df' contains a single numerical column 'Unix_Time' representing the timestamp

data = data.drop('Timestamp', axis = 1)


data.head()

# Engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}") #database

clean = pd.DataFrame(impute.transform(data), columns = data.columns).convert_dtypes()

clean1 = pd.DataFrame(winzor.transform(clean), columns = data.columns).convert_dtypes()

clean3 = pd.DataFrame(minmax.transform(clean1), columns = clean1.columns)

prediction = model1.predict(clean3)
prediction

# optimal_threshold=0.60
data["Clicked_on_Ad"] = np.zeros(len(prediction))

# taking threshold value and above the prob value will be treated as correct value 
data.loc[prediction > optimal_threshold, "Clicked_on_Ad"] = 1
data[['Clicked_on_Ad']] = data[['Clicked_on_Ad']].astype('int64')

data[['Clicked_on_Ad']] 
