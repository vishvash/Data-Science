'''Simple Linear regression
# Simple linear regression is a regression model that estimates the relationship
 between one independent variable and a dependent variable using a straight line.

# **Problem Statement**

# The Waist Circumference – Adipose Tissue Relationship:
 
# Studies have shown that individuals with excess Adipose tissue (AT) in 
their abdominal region have a higher risk of cardio-vascular diseases.
To assess the health conditions of a patient, doctor must get a report 
on the patients AT values. Computed Tomography, commonly called the CT Scan
is the only technique that allows for the precise and reliable measurement 
of the AT (at any site in the body). 

# 
# The problems with using the CT scan are:
# - Many physicians do not have access to this technology
# - Irradiation of the patient (suppresses the immune system)
# - Expensive
# 
# The Hospital/Organization wants to find an alternative solution for this 
problem, which can allow doctors to help their patients efficiently.
# 
'''

# CRISP-ML(Q) process model describes six phases:
# 
# - Business and Data Understanding
# - Data Preparation (Data Engineering)
# - Model Building (Machine Learning)
# - Model Evaluation and Tunning
# - Deployment
# - Monitoring and Maintenance
# 

'''
# **Objective(s):** Minimize the risk for patients
# or 
# Maximize the convience to doctors in assisting their patients
# 
# **Constraints:** CT Scan is the only option
# 
# **Research:** A group of researchers conducted a study with the aim of 
predicting abdominal AT area using simple anthropometric measurements, i.e., 
measurements on the human body
'''
 
# **Proposed Plan:**
# The Waist Circumference – Adipose Tissue data is a part of this study wherein
# the aim is to study how well waist circumference (WC) predicts the AT area
# 
# 
# **Benefits:**
# Is there a simpler yet reasonably accurate way to predict the AT area? i.e.,
# - Easily available
# - Risk free
# - Inexpensive
# 

# **Data Collection**
# Data: 
#     AT values from the historical Data
#     Waist Circumference of these patients.
# 
# Collection:
# 1. Evaluate the available Hospital records for relevant data (CT scan of patients)
# 
# 2. Record the Waist Circumference of patients - Primary Data
# 
# - Strategy to Collection Primary Data:
#     Call out the most recent patients (1 year old) with an offer of free 
#     consultation from a senior doctor to attract them to visit hospital.
#     Once the paitents visit the hospital, we can record their Waist 
#     Circumference with accuracy.

# # Explore the Patients Database (MySQL)

# Connect to the MySQL DB source for Primary data
# Load the datasets into Python dataxframe
import pandas as pd

data = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Simple Linear Regression/Simple Linear Regression/datasets/ATpatients.csv")
data2 = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Simple Linear Regression/Simple Linear Regression/datasets/waist.csv")

print(data.head())

print(data2.head())



from sqlalchemy import create_engine, text

engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user="root",# user
                               pw="1234", # passwrd
                               db="cardio")) #database

data.to_sql('atpatients',con = engine, if_exists = 'replace', index = False)

data2.to_sql('waist',con = engine, if_exists = 'replace', index = False)

# Declaring 'Patient' as a primary key in both tables
with engine.connect() as con:
    con.execute(text('ALTER TABLE `atpatients` ADD PRIMARY KEY (`Patient`);'))
    
with engine.connect() as con:
    con.execute(text('ALTER TABLE `waist` ADD PRIMARY KEY (`Patient`);'))


# Import only the required features into Python for Processing

sql = "SELECT A.Patient, A.AT, A.Sex, A.Age, B.Waist from atpatients as A Inner join waist as B on A.Patient = B.Patient;"


wcat_full = pd.read_sql_query(sql, engine)

wcat_full.head()

wcat_full.describe()

wcat_full.Sex.value_counts()

wcat_full.info()


wcat = wcat_full.drop(["Patient", "Sex", "Age"], axis = 1)

# Relevant fields for Regression Analysis
wcat.info()



# Importing necessary libraries
import pandas as pd # deals with data frame        # for Data Manipulation"
import numpy as np  # deals with numerical values  # for Mathematical calculations"
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from feature_engine.outliers import Winsorizer

from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# ############ optional ###############
wcat = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Simple Linear Regression/Simple Linear Regression/datasets/wc-at.csv")

wcat.info()

# ############ optional ###############

# EDA
wcat.describe()

wcat.head(10)

wcat.sort_values('Waist', ascending = True, inplace = True)

wcat.reset_index(inplace = True, drop = True)
wcat.head(10)

# Split the data into Target and Predictors
X = pd.DataFrame(wcat['Waist'])
Y = pd.DataFrame(wcat['AT'])

# Select numeric features for data preprocessing
numeric_features = ['Waist']

# Multiple boxplots in a single visualization.
# Columns with larger scales affect other columns. 
# Below code ensures each column gets its own y-axis.

# pandas plot() function with parameters kind = 'box' and subplots = True

wcat.plot(kind = 'box', subplots = True, sharey = False, figsize = (15, 8)) 
'''sharey True or 'all': x-axis or y-axis will be shared among all subplots.
False or 'none': each subplot x- or y-axis will be independent.'''
# increase spacing between subplots
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()


winsor = Winsorizer(capping_method = 'iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 1.5,
                          variables = numeric_features)

winsor


num_pipeline = Pipeline(steps = [('impute', SimpleImputer(strategy = 'mean'))])
outlier_pipeline = Pipeline(steps = [('winsor', winsor)])

num_pipeline

outlier_pipeline


preprocessor = ColumnTransformer(transformers = [('num', num_pipeline, numeric_features)])
print(preprocessor)

preprocessor1 = ColumnTransformer(transformers = [('wins', outlier_pipeline, numeric_features)])
print(preprocessor1)


impute_data = preprocessor.fit(X)
wcat['Waist'] = pd.DataFrame(impute_data.transform(X))

X2 = pd.DataFrame(wcat['Waist'])
winz_data = preprocessor1.fit(X2)

wcat['Waist'] = pd.DataFrame(winz_data.transform(X))


wcat.head(10)


# Save the data preprocessing pipelines
joblib.dump(impute_data, 'meanimpute')

joblib.dump(winz_data, 'winzor')


wcat.plot(kind = 'box', subplots = True, sharey = False, figsize = (15, 8)) 
'''sharey True or 'all': x- or y-axis will be shared among all subplots.
False or 'none': each subplot x- or y-axis will be independent.'''
# Increase spacing between subplots
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()

# Graphical Representation
plt.bar(height = wcat.AT, x = np.arange(1, 110, 1))

plt.hist(wcat.AT) #histogram

plt.bar(height = wcat.Waist, x = np.arange(1, 110, 1))

plt.hist(wcat.Waist)


# The above are manual approach to perform Exploratory Data Analysis (EDA). The alternate approach is to Automate the EDA process using Python libraries.
# 
# Auto EDA libraries:
# - Sweetviz
# - dtale
# - pandas profiling
# - autoviz

# 
# # **Automating EDA with Sweetviz:**
# 

# Using sweetviz to automate EDA is pretty simple and straight forward. 3 simple steps will provide a detailed report in html page.
# 
# step 1. Install sweetviz package using pip.
# - !pip install sweetviz
# 
# step2. import sweetviz package and call analyze function on the dataframe.
# 
# step3. Display the report on a html page created in the working directory with show_html function.
# 
import sweetviz as sv

# Analyzing the dataset
report = sv.analyze(wcat)

# Display the report
# report.show_notebook()  # integrated report in notebook

report.show_html('EDAreport.html') # html report generated in working directory



# # Bivariate Analysis
# Scatter plot
plt.scatter(x = wcat['Waist'], y = wcat['AT']) 

## Measure the strength of the relationship between two variables using Correlation coefficient.

np.corrcoef(wcat.Waist, wcat.AT)

# Covariance
cov_output = np.cov(wcat.Waist, wcat.AT)[0, 1]
cov_output

# wcat.cov()

dataplot = sns.heatmap(wcat.corr(), annot = True, cmap = "YlGnBu")


# # Linear Regression using statsmodels package
# Simple Linear Regression
model = smf.ols('AT ~ Waist', data = wcat).fit()

model.summary()

pred1 = model.predict(pd.DataFrame(wcat['Waist']))

pred1


# Regression Line
plt.scatter(wcat.Waist, wcat.AT)
plt.plot(wcat.Waist, pred1, "r")
plt.legend(['Observed data', 'Predicted line'])
plt.show()


# Error calculation (error = AV - PV)
res1 = wcat.AT - pred1

print(np.mean(res1))

res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1


# # Model Tuning with Transformations
# ## Log Transformation
# x = log(waist); y = at

plt.scatter(x = np.log(wcat['Waist']), y = wcat['AT'], color = 'brown')
np.corrcoef(np.log(wcat.Waist), wcat.AT) #correlation

model2 = smf.ols('AT ~ np.log(Waist)', data = wcat).fit()
model2.summary()


pred2 = model2.predict(pd.DataFrame(wcat['Waist']))

# Regression Line
plt.scatter(np.log(wcat.Waist), wcat.AT)
plt.plot(np.log(wcat.Waist), pred2, "r")
plt.legend(['Observed data', 'Predicted line'])
plt.show()

# Error calculation
res2 = wcat.AT - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


# ## Exponential transformation
# x = waist; y = log(at)

plt.scatter(x = wcat['Waist'], y = np.log(wcat['AT']), color = 'orange')
np.corrcoef(wcat.Waist, np.log(wcat.AT)) #correlation

model3 = smf.ols('np.log(AT) ~ Waist', data = wcat).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(wcat['Waist']))

# Regression Line
plt.scatter(wcat.Waist, np.log(wcat.AT))
plt.plot(wcat.Waist, pred3, "r")
plt.legend(['Observed data', 'Predicted line'])
plt.show()


pred3_at = np.exp(pred3)
print(pred3_at)

res3 = wcat.AT - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


# ## Polynomial transformation 
# x = waist; x^2 = waist*waist; y = log(at)

X = pd.DataFrame(wcat['Waist'])
# X.sort_values(by = ['Waist'], axis = 0, inplace = True)

Y = pd.DataFrame(wcat['AT'])


model4 = smf.ols('np.log(AT) ~ Waist + I(Waist*Waist)', data = wcat).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(wcat))
print(pred4)


plt.scatter(X['Waist'], np.log(Y['AT']))
plt.plot(X['Waist'], pred4, color = 'red')
plt.plot(X['Waist'], pred3, color = 'green', label = 'linear')
plt.legend(['Transformed Data', 'Polynomial Regression Line', 'Linear Regression Line'])
plt.show()

pred4_at = np.exp(pred4)
pred4_at

# Error calculation
res4 = wcat.AT - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4


# ### Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)

table_rmse

# # Evaluate the best model
# Data Split
train, test = train_test_split(wcat, test_size = 0.2, random_state = 0)

plt.scatter(train.Waist, np.log(train.AT))

plt.figure(2)
plt.scatter(test.Waist, np.log(test.AT))

# Fit the best model on train data
finalmodel = smf.ols('np.log(AT) ~ Waist + I(Waist*Waist)', data = train).fit()

# Trail and error by vishva
# poly_model = make_pipeline(PolynomialFeatures(degree = 2), LinearRegression())
# poly_model.fit(train[['Waist']], train[['AT']])
# test_pred = poly_model.predict(test[['Waist']])
# test_res = test.AT -  pd.Series(test_pred.flatten())
# test_sqrs = test_res * test_res
# test_mse = np.mean(test_sqrs)
# test_rmse = np.sqrt(test_mse)

# test_rmse


# Predict on test data
test_pred = finalmodel.predict(test)
pred_test_AT = np.exp(test_pred)

# Model Evaluation on Test data
test_res = test.AT - pred_test_AT
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)

test_rmse

# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_AT = np.exp(train_pred)
pred_train_AT

# Model Evaluation on train data
train_res = train.AT - pred_train_AT
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)

train_rmse

##### Save the Best model (Polynomial with 2-degree model) for Pipelining

poly_model = make_pipeline(PolynomialFeatures(degree = 2), LinearRegression())
poly_model.fit(wcat[['Waist']], wcat[['AT']])

pickle.dump(poly_model, open('poly_model.pkl', 'wb'))


### testing on new data
# Load the saved pipelines

impute = joblib.load('meanimpute')
winsor = joblib.load('winzor')
poly_model = pickle.load(open('poly_model.pkl', 'rb'))


wcat_test = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Simple Linear Regression/Simple Linear Regression/datasets/wc-at_test.csv")

clean1 = pd.DataFrame(impute.transform(wcat_test), columns = wcat_test.select_dtypes(exclude = ['object']).columns)

clean2 = pd.DataFrame(winsor.transform(clean1), columns = clean1.columns)

prediction = pd.DataFrame(poly_model.predict(clean2), columns = ['Pred_AT'])

final = pd.concat([prediction, wcat_test], axis = 1)

final

