## Problem Statement
'''
An analytics company has been tasked with the crucial job of finding out what factors affect a startup company and if it will be profitable or not. For this, they have collected some historical data and would like to apply multilinear regression to derive brief insights into their data. Predict profit, given different attributes for various startup companies.


**Linear regression**
Linear regression is a ML model that estimates the relationship between 
independent variables and a dependent variable using a linear equation 
(straight line equation) in a multidimensional space.

**CRISP-ML(Q) process model describes six phases:**

- Business and Data Understanding
- Data Preparation (Data Engineering)
- Model Building (Machine Learning)
- Model Evaluation and Tunning
- Deployment
- Monitoring and Maintenance


**Objective(s):** Maximize the profits

**Constraints:** Minimize the spends

**Success Criteria**

- **Business Success Criteria**: Improve the profits from anywhere between 10% to 20%

- **ML Success Criteria**: RMSE should be less than 0.15

- **Economic Success Criteria**: Business should see increase in sales by atleast 20%
'''

# Load the Data and perform EDA and Data Preprocessing

# Importing necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# pip install sidetable
# pip install watchdog
# pip install --upgrade watchdog

import sidetable

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from feature_engine.outliers import Winsorizer

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from sklearn.model_selection import train_test_split
# import statsmodels.formula.api as smf
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import joblib
import pickle

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

# Recursive feature elimination
from sklearn.feature_selection import RFE
from sqlalchemy import create_engine, text


engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user = "root",# user
                               pw = "1234", # passwrd
                               db = "profit_db")) #database


# Load the offline data into Database to simulate client conditions
profit = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Multiple_Linear Regression/Assignments/Multiple Linear Regression/50_Startups.csv")
profit.to_sql('profit', con = engine, if_exists = 'replace', chunksize = 1000, index= False)


#### Read the Table (data) from MySQL database

sql = 'SELECT * FROM profit'
# sql2="show tables"
# tables = pd.read_sql_query(sql2, engine)

dataset = pd.read_sql_query(text(sql), engine.connect())


# dataset = pd.read_csv(r"C:/Data/profitwithState.csv")

#### Descriptive Statistics and Data Distribution
dataset.describe()

# Missing values check
dataset.isnull().any()
dataset.info()


# Seperating input and output variables 
X = pd.DataFrame(dataset.iloc[:, 1:5])
y = pd.DataFrame(dataset.iloc[:, 0])

# Checking for unique values
X["State"].unique()

X["State"].value_counts()

# Build a frequency table using sidetable library
X.stb.freq(["State"])

# Segregating Non-Numeric features
categorical_features = X.select_dtypes(include = ['object']).columns
print(categorical_features)


# Segregating Numeric features
numeric_features = X.select_dtypes(exclude = ['object']).columns
print(numeric_features)


## Missing values Analysis
# Define pipeline for missing data if any

num_pipeline = Pipeline(steps = [('impute', SimpleImputer(strategy = 'mean'))])

preprocessor = ColumnTransformer(transformers = [('num', num_pipeline, numeric_features)])

# Fit the imputation pipeline to input features
imputation = preprocessor.fit(X)

# Save the pipeline
joblib.dump(imputation, 'meanimpute')

# Transformed data
cleandata = pd.DataFrame(imputation.transform(X), columns = numeric_features)
cleandata


## Outlier Analysis

# Multiple boxplots in a single visualization.
# Columns with larger scales affect other columns. 
# Below code ensures each column gets its own y-axis.

# pandas plot() function with parameters kind = 'box' and subplots = True

X.plot(kind = 'box', subplots = True, sharey = False, figsize = (25, 18)) 
'''sharey True or 'all': x- or y-axis will be shared among all subplots.
False or 'none': each subplot x- or y-axis will be independent.'''
# Increase spacing between subplots
plt.subplots_adjust(wspace = 0.75) 
# ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()


# Winsorization for outlier treatment
winsor = Winsorizer(capping_method = 'iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 1.5,
                          variables = list(cleandata.columns))

clean = winsor.fit(cleandata)

# Save winsorizer model
joblib.dump(clean, 'winsor')

cleandata1 = pd.DataFrame(clean.transform(cleandata), columns = numeric_features)

# Boxplot
cleandata1.plot(kind = 'box', subplots = True, sharey = False, figsize = (25, 18)) 
plt.subplots_adjust(wspace = 0.75)
plt.show()


# Scaling
## Scaling with MinMaxScaler
scale_pipeline = Pipeline([('scale', MinMaxScaler())])

scale_columntransfer = ColumnTransformer([('scale', scale_pipeline, numeric_features)]) 
# Skips the transformations for remaining columns

scale = scale_columntransfer.fit(cleandata1)

# Save Minmax scaler pipeline model
joblib.dump(scale, 'minmax')

scaled_data = pd.DataFrame(scale.transform(cleandata1), columns = numeric_features)
scaled_data.describe()



## Encoding
# Categorical features
encoding_pipeline = Pipeline([('onehot', OneHotEncoder())])

preprocess_pipeline = ColumnTransformer([('categorical', encoding_pipeline, categorical_features)])

clean =  preprocess_pipeline.fit(X)   # Works with categorical features only

# Save the encoding model
joblib.dump(clean, 'encoding')

encode_data = pd.DataFrame(clean.transform(X))


# To get feature names for Categorical columns after Onehotencoding 
encode_data.columns = clean.get_feature_names_out(input_features = X.columns)
encode_data.info()



clean_data = pd.concat([scaled_data, encode_data], axis = 1)  
# concatenated data will have new sequential index
clean_data.info()


####################
# Multivariate Analysis
sns.pairplot(dataset)   # original data

# Correlation Analysis on Original Data
orig_df_cor = dataset.corr(numeric_only=True)
orig_df_cor



# Heatmap
dataplot = sns.heatmap(orig_df_cor, annot = True, cmap = "YlGnBu")

# Heatmap enhanced
# Generate a mask to show values on only the bottom triangle
# Upper triangle of an array.
mask = np.triu(np.ones_like(orig_df_cor, dtype = bool))
sns.heatmap(orig_df_cor, annot = True, mask = mask, vmin = -1, vmax = 1)
plt.title('Correlation Coefficient Of Predictors')
plt.show()


# Library to call OLS model
# import statsmodels.api as sm

# Build a vanilla model on full dataset

# By default, statsmodels fits a line passing through the origin, i.e. it 
# doesn't fit an intercept. Hence, you need to use the command 'add_constant' 
# so that it also fits an intercept

P = add_constant(clean_data)

basemodel = sm.OLS(y, P).fit()
basemodel.summary()

# p-values of coefficients found to be insignificant due to colinearity

# Identify the variale with highest colinearity using Variance Inflation factor (VIF)
# Variance Inflation Factor (VIF)
# Assumption: VIF > 10 = colinearity
# VIF on clean Data
vif = pd.Series([variance_inflation_factor(P.values, i) for i in range(P.shape[1])], index = P.columns)
vif
# inf = infinity


# Tune the model by verifying for influential observations
sm.graphics.influence_plot(basemodel)

clean_data1_new = clean_data.drop(clean_data.index[[45, 46, 48, 49]])
y_new = y.drop(y.index[[45, 46, 48, 49]])

# clean_data1_new.drop("const", axis=1, inplace=True)

# Build model on dataset
basemode3 = sm.OLS(y_new, clean_data1_new).fit()
basemode3.summary()



# Splitting data into training and testing data set
X_train, X_test, Y_train, Y_test = train_test_split(clean_data1_new, y_new, 
                                                    test_size = 0.2, random_state = 0) 

## Build the best model Model building with out cv
model = sm.OLS(Y_train, X_train).fit()
model.summary()

# Predicting upon X_train
ytrain_pred = model.predict(X_train)
r_squared_train = r2_score(Y_train, ytrain_pred)
r_squared_train

# Train residual values
train_resid  = Y_train.Profit - ytrain_pred
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse


# Predicting upon X_test
y_pred = model.predict(X_test)

# checking the Accurarcy by using r2_score
r_squared = r2_score(Y_test, y_pred)
r_squared

# Test residual values
test_resid  = Y_test.Profit - y_pred
# RMSE value for train data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


## Scores with Cross Validation (cv)
# k-fold CV (using all variables)
lm = LinearRegression()

## Scores with KFold
folds = KFold(n_splits = 5, shuffle = True, random_state = 100)

scores = cross_val_score(lm, X_train, Y_train, scoring = 'r2', cv = folds)
scores   



## Model building with CV and RFE

# step-1: create a cross-validation scheme
folds = KFold(n_splits = 5, shuffle = True, random_state = 100)

# step-2: specify range of hyperparameters to tune
hyper_params = [{'n_features_to_select': list(range(1, 9))}]


# step-3: perform grid search
# 3.1 specify model
# lm = LinearRegression()
lm.fit(X_train, Y_train)

# Recursive Feature Elimination
rfe = RFE(lm)

# 3.2 call GridSearchCV()
model_cv = GridSearchCV(estimator = rfe, 
                        param_grid = hyper_params, 
                        scoring = 'r2', 
                        cv = folds, 
                        verbose = 1,
                        return_train_score = True)      

# fit the model
model_cv.fit(X_train, Y_train)     

cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results

# plotting cv results
plt.figure(figsize = (16, 6))

plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_train_score"])
plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_test_score"])
plt.xlabel('Number of Features')
plt.ylabel('r-squared')
plt.title("Optimal Number of Features")
plt.legend(['train score', 'test score'], loc = 'upper left')

# train and test scores get stable after 3rd feature. 
# we can select number of optimal features more than 3

model_cv.best_params_

cv_lm_grid = model_cv.best_estimator_
cv_lm_grid

## Saving the model into pickle file
pickle.dump(cv_lm_grid, open('Profit.pkl', 'wb'))

## Testing
data = pd.read_excel(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Multiple_Linear Regression/Assignments/Multiple Linear Regression/profit_test.xlsx")

model1 = pickle.load(open('Profit.pkl','rb'))
impute = joblib.load('meanimpute')
winsor = joblib.load('winsor')
minmax = joblib.load('minmax')
encoding = joblib.load('encoding')

clean = pd.DataFrame(impute.transform(data), columns = data.select_dtypes(exclude = ['object']).columns)

clean1 = pd.DataFrame(winsor.transform(clean),columns = data.select_dtypes(exclude = ['object']).columns)
clean2 = pd.DataFrame(minmax.transform(clean1),columns = data.select_dtypes(exclude = ['object']).columns)
clean3 = pd.DataFrame(encoding.transform(data), columns = encoding.get_feature_names_out(input_features = data.columns))

clean_data = pd.concat([clean2, clean3], axis = 1)
clean_data.info()
# clean_data1 = clean_data.drop(clean_data[['WT']], axis = 1)
# clean_data1 = clean_data.drop(3, axis = 1)

prediction = pd.DataFrame(model1.predict(clean_data), columns = ['Profit_pred'])

prediction
