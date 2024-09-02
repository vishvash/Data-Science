'''Simple Linear regression
# Simple linear regression is a regression model that estimates the relationship
 between one independent variable and a dependent variable using a straight line.

# **Problem Statement**

# A logistics company recorded the time taken for delivery and the time taken for the sorting of the items for delivery. Build a Simple Linear Regression model to find the relationship between delivery time and sorting time with the delivery time as the target variable. Apply necessary transformations and record the RMSE and correlation coefficient values for different models
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
# **Objective(s):** Minimize Delivery Time
# 
# **Constraints:** Operational or human resource constraints

'''
 
# Connect to the MySQL DB source for Primary data
# Load the datasets into Python dataxframe
import pandas as pd

data = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Simple Linear Regression/Assignments/Simple LinearReg/delivery_time.csv")

print(data.head())



from sqlalchemy import create_engine, text

engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user="root",# user
                               pw="1234", # passwrd
                               db="delivery_db")) #database

data.to_sql('delivery',con = engine, if_exists = 'replace', index = False)



# Import only the required features into Python for Processing

sql = "SELECT * from delivery;"


delv = pd.read_sql_query(sql, engine)

delv.head()

delv.describe()

delv.info()


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



delv.sort_values('Sorting_Time', ascending = True, inplace = True)

delv.reset_index(inplace = True, drop = True)
delv.head(10)

# Split the data into Target and Predictors
X = pd.DataFrame(delv['Sorting_Time'])
Y = pd.DataFrame(delv['Delivery_Time'])

# Select numeric features for data preprocessing
numeric_features = ['Sorting_Time']

# Multiple boxplots in a single visualization.
# Columns with larger scales affect other columns. 
# Below code ensures each column gets its own y-axis.

# pandas plot() function with parameters kind = 'box' and subplots = True

delv.plot(kind = 'box', subplots = True, sharey = False, figsize = (15, 8)) 
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
delv['Sorting_Time'] = pd.DataFrame(impute_data.transform(X))

X2 = pd.DataFrame(delv['Sorting_Time'])
winz_data = preprocessor1.fit(X2)

delv['Sorting_Time'] = pd.DataFrame(winz_data.transform(X2))


delv.head(10)


# Save the data preprocessing pipelines
joblib.dump(impute_data, 'meanimpute')

joblib.dump(winz_data, 'winzor')


delv.plot(kind = 'box', subplots = True, sharey = False, figsize = (15, 8)) 
'''sharey True or 'all': x- or y-axis will be shared among all subplots.
False or 'none': each subplot x- or y-axis will be independent.'''
# Increase spacing between subplots
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()

# Graphical Representation
plt.bar(height = delv.Delivery_Time, x = np.arange(1, 22, 1))

plt.hist(delv.Delivery_Time) #histogram

plt.bar(height = delv.Sorting_Time, x = np.arange(1, 22, 1))

plt.hist(delv.Sorting_Time)


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
report = sv.analyze(delv)

# Display the report
# report.show_notebook()  # integrated report in notebook

report.show_html('EDAreport.html') # html report generated in working directory



# # Bivariate Analysis
# Scatter plot
plt.scatter(x = delv['Sorting_Time'], y = delv['Delivery_Time']) 

## Measure the strength of the relationship between two variables using Correlation coefficient.

np.corrcoef(delv.Sorting_Time, delv.Delivery_Time)

# Covariance
cov_output = np.cov(delv.Sorting_Time, delv.Delivery_Time)[0, 1]
cov_output

# delv.cov()

dataplot = sns.heatmap(delv.corr(), annot = True, cmap = "YlGnBu")


# # Linear Regression using statsmodels package
# Simple Linear Regression
model = smf.ols('Delivery_Time ~ Sorting_Time', data = delv).fit()

model.summary()

pred1 = model.predict(pd.DataFrame(delv['Sorting_Time']))

pred1


# Regression Line
plt.scatter(delv.Sorting_Time, delv.Delivery_Time)
plt.plot(delv.Sorting_Time, pred1, "r")
plt.legend(['Observed data', 'Predicted line'])
plt.show()


# Error calculation (error = AV - PV)
res1 = delv.Delivery_Time - pred1

print(np.mean(res1))

res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1


# # Model Tuning with Transformations
# ## Log Transformation
# x = log(Sorting_Time); y = Delivery_Time

plt.scatter(x = np.log(delv['Sorting_Time']), y = delv['Delivery_Time'], color = 'brown')
np.corrcoef(np.log(delv.Sorting_Time), delv.Delivery_Time) #correlation

model2 = smf.ols('Delivery_Time ~ np.log(Sorting_Time)', data = delv).fit()
model2.summary()


pred2 = model2.predict(pd.DataFrame(delv['Sorting_Time']))

# Regression Line
plt.scatter(np.log(delv.Sorting_Time), delv.Delivery_Time)
plt.plot(np.log(delv.Sorting_Time), pred2, "r")
plt.legend(['Observed data', 'Predicted line'])
plt.show()

# Error calculation
res2 = delv.Delivery_Time - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


# ## Exponential transformation
# x = Sorting_Time; y = log(Delivery_Time)

plt.scatter(x = delv['Sorting_Time'], y = np.log(delv['Delivery_Time']), color = 'orange')
np.corrcoef(delv.Sorting_Time, np.log(delv.Delivery_Time)) #correlation

model3 = smf.ols('np.log(Delivery_Time) ~ Sorting_Time', data = delv).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(delv['Sorting_Time']))

# Regression Line
plt.scatter(delv.Sorting_Time, np.log(delv.Delivery_Time))
plt.plot(delv.Sorting_Time, pred3, "r")
plt.legend(['Observed data', 'Predicted line'])
plt.show()


pred3_at = np.exp(pred3)
print(pred3_at)

res3 = delv.Delivery_Time - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


# ## Polynomial transformation 
# x = Sorting_Time; x^2 = Sorting_Time*Sorting_Time; y = log(Delivery_Time)

X = pd.DataFrame(delv['Sorting_Time'])
# X.sort_values(by = ['Sorting_Time'], axis = 0, inplace = True)

Y = pd.DataFrame(delv['Delivery_Time'])


model4 = smf.ols('np.log(Delivery_Time) ~ Sorting_Time + I(Sorting_Time*Sorting_Time)', data = delv).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(delv))
print(pred4)


plt.scatter(X['Sorting_Time'], np.log(Y['Delivery_Time']))
plt.plot(X['Sorting_Time'], pred4, color = 'red')
plt.plot(X['Sorting_Time'], pred3, color = 'green', label = 'linear')
plt.legend(['Transformed Data', 'Polynomial Regression Line', 'Linear Regression Line'])
plt.show()

pred4_at = np.exp(pred4)
pred4_at

# Error calculation
res4 = delv.Delivery_Time - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4


# ## Polynomial transformation with 3 degrees
# x = Sorting_Time; x^2 = Sorting_Time*Sorting_Time; y = log(Delivery_Time)

X = pd.DataFrame(delv['Sorting_Time'])
# X.sort_values(by = ['Sorting_Time'], axis = 0, inplace = True)

Y = pd.DataFrame(delv['Delivery_Time'])


model5 = smf.ols('np.log(Delivery_Time) ~ Sorting_Time + I(Sorting_Time**2) + I(Sorting_Time**3)', data = delv).fit()
model5.summary()

pred5 = model5.predict(pd.DataFrame(delv))
print(pred5)


plt.scatter(X['Sorting_Time'], np.log(Y['Delivery_Time']))
plt.plot(X['Sorting_Time'], pred5, color = 'blue')
# plt.plot(X['Sorting_Time'], pred4, color = 'red')
plt.plot(X['Sorting_Time'], pred3, color = 'green', label = 'linear')
plt.legend(['Transformed Data', 'Polynomial with 3 degrees', 'Linear Regression Line'])
plt.show()

pred5_at = np.exp(pred5)
pred5_at

# Error calculation
res5 = delv.Delivery_Time - pred5_at
res_sqr5 = res5 * res5
mse5 = np.mean(res_sqr5)
rmse5 = np.sqrt(mse5)
rmse5


# ### Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model", "Poly 3 deg"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4, rmse5])}
table_rmse = pd.DataFrame(data)

table_rmse

# # Evaluate the best model
# Data Split
train, test = train_test_split(delv, test_size = 0.2, random_state = 0)

plt.scatter(train.Sorting_Time, np.log(train.Delivery_Time))

plt.figure(2)
plt.scatter(test.Sorting_Time, np.log(test.Delivery_Time))

# Fit the best model on train data
finalmodel = smf.ols('np.log(Delivery_Time) ~ Sorting_Time + I(Sorting_Time**2) + I(Sorting_Time**3)', data = train).fit()

# Trail and error by vishva
# poly_model = make_pipeline(PolynomialFeatures(degree = 2), LinearRegression())
# poly_model.fit(train[['Sorting_Time']], train[['Delivery_Time']])
# test_pred = poly_model.predict(test[['Sorting_Time']])
# test_res = test.Delivery_Time -  pd.Series(test_pred.flatten())
# test_sqrs = test_res * test_res
# test_mse = np.mean(test_sqrs)
# test_rmse = np.sqrt(test_mse)

# test_rmse


# Predict on test data
test_pred = finalmodel.predict(test)
pred_test_AT = np.exp(test_pred)

# Model Evaluation on Test data
test_res = test.Delivery_Time - pred_test_AT
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)

test_rmse

# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_AT = np.exp(train_pred)
pred_train_AT

# Model Evaluation on train data
train_res = train.Delivery_Time - pred_train_AT
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)

train_rmse

##### Save the Best model (Polynomial with 2-degree model) for Pipelining

poly_model = make_pipeline(PolynomialFeatures(degree = 3), LinearRegression())
poly_model.fit(delv[['Sorting_Time']], delv[['Delivery_Time']])

pickle.dump(poly_model, open('poly_model.pkl', 'wb'))


### testing on new data
# Load the saved pipelines

impute = joblib.load('meanimpute')
winsor = joblib.load('winzor')
poly_model = pickle.load(open('poly_model.pkl', 'rb'))


delv_test = pd.read_excel(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Simple Linear Regression/Assignments/Simple LinearReg/Sortingtime_test.xlsx")

clean1 = pd.DataFrame(impute.transform(delv_test), columns = delv_test.select_dtypes(exclude = ['object']).columns)

clean2 = pd.DataFrame(winsor.transform(clean1), columns = clean1.columns)

prediction = pd.DataFrame(poly_model.predict(clean2), columns = ['Pred_Delivery_Time'])

final = pd.concat([prediction, delv_test], axis = 1)

final

