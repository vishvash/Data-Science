'''
Problem Statement:
    1.	You work for a consumer finance company that specializes in lending loans to urban customers. When the company receives a loan application, the company has to make a decision for loan approval based on the applicant’s profile. Two types of risks are associated with the bank’s decision: 
•	If the applicant is likely to repay the loan, then not approving the loan results in a loss of business to the company 
•	If the applicant is not likely to repay the loan, i.e. he/she is likely to default, then approving the loan may lead to a financial loss for the company 

The data given below contains information about past loan applicants and whether they ‘defaulted’4 or not. The aim is to identify patterns that indicate if a person is likely to default, which may be used for taking actions such as denying the loan, reducing the amount of the loan, lending (for risky applicants) at a higher interest rate, etc. 

In this case study, you will use EDA to understand how consumer attributes and loan attributes influence the tendency of default. 

When a person applies for a loan, there are two types of decisions that could be taken by the company: 

1. Loan accepted: If the company approves the loan, there are 3 possible scenarios described below: 
•	Fully paid: Applicant has fully paid the loan (the principal and the interest rate) 
•	Current: Applicant is in the process of paying the installments, i.e., the tenure of the loan is not yet completed. These candidates are not labeled as 'defaulted'. 
•	Charged-off: Applicant has not paid the installments in due time for a long period of time, i.e. he/she has defaulted on the loan  
2. Loan rejected: The company had rejected the loan (because the candidate does not meet their requirements etc.). Since the loan was rejected, there is no transactional history of those applicants with the company and so this data is not available with the company (and thus in this dataset)

Like most other lending companies, lending loans to ‘risky’ applicants is the largest source of financial loss (called credit loss). Credit loss is the amount of money lost by the lender when the borrower refuses to pay or runs away with the money owed. In other words, borrowers who default cause the largest amount of loss to the lenders. In this case, the customers labeled as 'charged-off' are the 'defaulters'.  
If one can identify these risky loan applicants, then such loans can be reduced thereby cutting down the amount of credit loss. 
In other words, the company wants to understand the driving factors (or driver variables) behind loan default, i.e. the variables which are strong indicators of default.  The company can utilize this knowledge for its portfolio and risk assessment.  

Perform Multinomial regression on the dataset in which loan_status is the output (Y) variable and it has three levels in it. 

Business objective: Minimize the defaulters
Business constraints: Identify the repayabel amount by defaulters

Business success criteria: Reduce the proportion of loan defaults to less than 5%
ML success criteria: Achieve the accurace of at least 70 percent
Economic success criteria: Increaset the return on investment atleast by 10%

'''

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


mode = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Multinomial Regression/Assignments/Multinomial Regression Assignment/loan_refined.csv").convert_dtypes()
mode = mode.head(1000)
mode.head(10)
mode.info()
mode.describe()
mode.loan_status.value_counts()

mode.drop(columns = 'collection_recovery_fee', inplace = True) #due to high outliers


mode.boxplot(by = "loan_status" , sharey = False, figsize = (15, 8))
plt.subplots_adjust(wspace=0.5,  hspace=0.5) 

# Scatter plot between each possible pair of independent variable and also histogram for each independent variable 
# sns.pairplot(mode, hue = "loan_status") # With showing the category of each car loan_status in the scatter plot
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
X = mode.iloc[:,1:]

# Target
Y = mode.iloc[:,[0]]

X.info()

# Numeric input features
numeric_features = X.select_dtypes(exclude = ['object']).columns

num_pipeline1 = Pipeline(steps = [('impute1', SimpleImputer(strategy = 'mean'))])                                                

# Imputation Transformer
preprocessor = ColumnTransformer([('mean', num_pipeline1, numeric_features)])

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
winsor = Winsorizer(capping_method = 'gaussian', # choose  IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 1,
                          variables = list(X1.columns))

outlier_pipeline = Pipeline(steps=[('winsor', winsor)])

preprocessor1 = ColumnTransformer(transformers = [('wins', outlier_pipeline, numeric_features)], remainder = 'passthrough')
print(preprocessor1)

# Train the pipeline
winz_data = preprocessor1.fit(X1)

# Save the pipeline
joblib.dump(winz_data, 'winzor')

# X2 = X1
# X1 = X2

X1 = pd.DataFrame(winz_data.transform(X1), columns=X1.columns)


# X1=winz_data.transform(X1)
X1.info()
X1.describe()

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

data = pd.read_excel(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Multinomial Regression/Assignments/Multinomial Regression Assignment/loan_test.xlsx")

data.drop(columns = 'collection_recovery_fee', inplace = True)

clean = pd.DataFrame(impute.transform(data), columns = data.columns)
clean1 = pd.DataFrame(winzor.transform(clean), columns = data.columns)
clean3 = pd.DataFrame(minmax.transform(clean1), columns = data.columns)

prediction = pd.DataFrame(model.predict(clean3), columns = ['loan_status'])
prediction

