import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split # train and test 

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from feature_engine.outliers import Winsorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score # confusion_matrix
import pickle, joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


mode = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Multinomial Regression/Multinomial Regression/mode.csv")
mode.head(10)
mode.describe()
mode.choice.value_counts()

# Boxplot of independent variable distribution for each category of choice 
sns.boxplot(x = "choice", y = "cost.car", data = mode)
sns.boxplot(x = "choice", y = "cost.carpool", data = mode)
sns.boxplot(x = "choice", y = "cost.bus", data = mode)
sns.boxplot(x = "choice", y = "cost.rail", data = mode)
sns.boxplot(x = "choice", y = "time.car", data = mode)
sns.boxplot(x = "choice", y = "time.bus", data = mode)
sns.boxplot(x = "choice", y = "time.rail", data = mode)

mode.boxplot(by = "choice" , sharey = False, figsize = (15, 8))

# mode.plot(kind = 'box', subplots = True, sharey = False, figsize = (15, 8), by = "choice") 

# Scatter plot for each categorical choice of car
sns.stripplot(x = "choice", y = "cost.car", jitter = True, data = mode)
sns.stripplot(x = "choice", y = "cost.carpool", jitter = True, data = mode)
sns.stripplot(x = "choice", y = "cost.bus", jitter = True, data = mode)
sns.stripplot(x = "choice", y = "cost.rail", jitter = True, data = mode)
sns.stripplot(x = "choice", y = "time.cars", jitter = True, data = mode)
sns.stripplot(x = "choice", y = "time.bus", jitter = True, data = mode)
sns.stripplot(x = "choice", y = "time.rail", jitter = True, data = mode)


# Scatter plot between each possible pair of independent variable and also histogram for each independent variable 
sns.pairplot(mode, hue = "choice") # With showing the category of each car choice in the scatter plot
# sns.pairplot(mode)

# Correlation values between each independent features
a = mode.iloc[:,1:].corr()

##########################
## Auto EDA ## 
import dtale
import pandas as pd

d = dtale.show(mode)
d.open_browser()

##########################

mode.info()

# Predictors
X = mode[['cost.car', 'cost.carpool', 'cost.bus', 'cost.rail','time.car', 'time.carpool', 'time.bus', 'time.rail']]

# Target
Y = mode[['choice']]

X.info()

# Numeric input features
numeric_features = X.select_dtypes(exclude = ['object']).columns

num_pipeline1 = Pipeline(steps = [('impute1', SimpleImputer(strategy = 'mean'))])                                                

# Imputation Transformer
preprocessor = ColumnTransformer([
        ('mean', num_pipeline1, numeric_features)])

print(preprocessor)

# Fit the data to pipeline
impute_data = preprocessor.fit(X)

# Save the pipeline
joblib.dump(impute_data, 'impute')

# Transform the data
X1 = pd.DataFrame(impute_data.transform(X), columns = X.columns)

X1.isna().sum()

# Outlier Analysis
# Multiple boxplots in a single visualization.
# pandas plot() function with parameters kind = 'box' and subplots = True
X1.plot(kind = 'box', subplots = True, sharey = False, figsize = (15, 8)) 

'''sharey True or 'all': x- or y-axis will be shared among all subplots.
False or 'none': each subplot x- or y-axis will be independent.'''

# increase spacing between subplots
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()


# from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method = 'iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 1.5,
                          variables = list(X1.columns))

outlier_pipeline = Pipeline(steps=[('winsor', winsor)])

preprocessor1 = ColumnTransformer(transformers = [('wins', outlier_pipeline, numeric_features)], remainder = 'passthrough')
print(preprocessor1)

# Train the pipeline
winz_data = preprocessor1.fit(X1)

# Save the pipeline
joblib.dump(winz_data, 'winzor')


X1[['cost.car', 'cost.carpool', 'cost.bus', 'cost.rail','time.car', 'time.carpool', 'time.bus', 'time.rail']]=winz_data.transform(X1)
X1.info()

# Boxplot
X1.plot(kind = 'box', subplots = True, sharey = False, figsize = (15, 8)) 
# increase spacing between subplots
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()

# Minmax Scaler
scale_pipeline = Pipeline(steps = [('scale', MinMaxScaler())])
preprocessor2 = ColumnTransformer(transformers = [('scale', scale_pipeline, numeric_features)], remainder = 'passthrough')
print(preprocessor2)

scale = preprocessor2.fit(X1)

# Save the pipeline
joblib.dump(scale, 'scale')

X2 = pd.DataFrame(scale.transform(X1), columns = X1.columns)
X2

# Data Partitioning
X_train, X_test, Y_train, Y_test = train_test_split(X2, Y, test_size = 0.2, 
                                                    random_state = 0,
                                                    stratify = Y)

# ‘multinomial’ option is supported only by the ‘lbfgs’ and ‘newton-cg’ solvers
logmodel = LogisticRegression(multi_class = "multinomial", solver = "newton-cg")

# Train the model
model = logmodel.fit(X_train, Y_train)

# Train accuracy 
train_predict = model.predict(X_train) # Train predictions 

accuracy_score(Y_train, train_predict) 


# Predict the results for Test Data
test_predict = model.predict(X_test) # Test predictions

# Test accuracy 
accuracy_score(Y_test, test_predict)


# Hyperparameter Optimization
logmodel1 = LogisticRegression(multi_class = "multinomial")

param_grid = [    
    {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
    'C' : np.logspace(-4, 4, 20),
    'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
    'max_iter' : [100, 1000,2500, 5000]
    }
]


# from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(logmodel1, param_grid = param_grid, cv = 3, verbose=True, n_jobs=-1)
best_clf = clf.fit(X_train, Y_train)

# Best estimator
best_clf.best_estimator_

print (f'Accuracy - : {best_clf.score(X_train, Y_train):.3f}')

print (f'Accuracy - : {best_clf.score(X_test, Y_test):.3f}')

# Y1 = np.ravel(Y)

# Fitting on Full data
best_clf1 = clf.fit(X2, Y)

best_clf1.best_estimator_

print (f'Accuracy - : {best_clf1.score(X2, Y):.3f}')
print (f'Accuracy - : {best_clf1.score(X_test, Y_test):.3f}')

# Save the best Model
pickle.dump(best_clf1, open('multinomial.pkl', 'wb'))


############################
# Predictions on New Data

model = pickle.load(open('multinomial.pkl', 'rb'))
impute = joblib.load('impute')
winzor = joblib.load('winzor')
minmax = joblib.load('scale')

data = pd.read_excel(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Multinomial Regression/Multinomial Regression/mode_test.xlsx")

clean = pd.DataFrame(impute.transform(data), columns = data.columns)
clean1 = pd.DataFrame(winzor.transform(clean), columns = data.columns)
clean3 = pd.DataFrame(minmax.transform(clean1), columns = data.columns)

prediction = pd.DataFrame(model.predict(clean3), columns = ['choice'])
prediction

