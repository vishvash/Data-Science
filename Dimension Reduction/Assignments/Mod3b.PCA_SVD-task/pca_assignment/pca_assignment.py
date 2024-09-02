# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 18:02:44 2024

@author: Lenovo
"""

'''

Problem Statements:
    
The average retention rate in the insurance industry is 84%, with the top-performing agencies in the 93% - 95% range. 

Retaining customers is all about the long-term relationship you build. 

Offering a discount on the client’s current policy will ensure he/she buys a new product or renews the current policy. 

Studying clients' purchasing behaviour to determine which products they're most likely to buy is essential. 

The insurance company wants to analyze their customer’s behaviour to strategies offers to increase customer loyalty.

CRISP-ML(Q) process model describes six phases:
1. Business and Data Understanding
2. Data Preparation
3. Model Building
4. Model Evaluation
5. Deployment
6. Monitoring and Maintenance

Objective: Maximize the Sales 
Constraints: Minimize the Customer Retention
Success Criteria: 
Business Success Criteria: Increase the Sales by 10% to 12% by targeting cross-selling opportunities on current customers.
ML Success Criteria: NA
Economic Success Criteria: The insurance company will see an increase in revenues by at least 8% 


'''

# #### Install the required packages if not available

# !pip install feature_engine
# !pip install dtale


# **Importing required packages**


import numpy as np
import pandas as pd
import sweetviz
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.decomposition import PCA
from kneed import KneeLocator

from sqlalchemy import create_engine, text

user = 'root' # user name
pw = '1234' # password
db = 'insur_db' # database
# creating engine to connect database
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# **Import the data**
insur = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Dimension Reduction/Assignments/key/AutoInsurance (1).csv")
insur
# dumping data into database 
# name should be in lower case
insur.to_sql('insur_clustering', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

# loading data from database
sql = text('select * from insur_clustering')

df = pd.read_sql_query(sql, con = engine.connect())

print(df)

df.info()

# # EXPLORATORY DATA ANALYSIS (EDA) / DESCRIPTIVE STATISTICS
# ***Descriptive Statistics and Data Distribution Function***
res = df.describe()

# Handle duplicates
df.Customer.duplicated().sum()

# Filter the numerical columns
df1 = df.select_dtypes(exclude=['object'])

# AutoEDA
# Automated Libraries
# import sweetviz
my_report = sweetviz.analyze([df1, "df1"])
my_report.show_html('Report.html')

# Data Preprocessing
# Checking Null Values
df1.isnull().sum()

# PCA can be implemented only on Numeric features
df1.info()
numeric_features = df1.select_dtypes(exclude = ['object']).columns
numeric_features

# Define the Pipeline steps
# Define PCA model
pca = PCA(n_components = 8)

# Make Pipeline

# **By using mean imputation, null values can be imputed**
# **Data has to be standardized to address the scale difference**

num_pipeline = make_pipeline(SimpleImputer(strategy = 'mean'), StandardScaler(), pca)
num_pipeline

# Pass the raw data through pipeline
processed = num_pipeline.fit(df1[numeric_features]) 
processed


# Save the End to End PCA pipeline with Imputation and Standardization
import joblib
joblib.dump(processed, 'PCA_DimRed')

import os
os.getcwd()

# Import the pipeline
model = joblib.load("PCA_DimRed")
model

# ## Apply the saved model on to the Dataset to extract PCA values
pca_res = pd.DataFrame(model.transform(df1[numeric_features]))
pca_res

# PCA weights
model['pca'].components_

# Take a closer look at the components
components = pd.DataFrame(model['pca'].components_, columns = numeric_features).T
components.columns = ['pc0', 'pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6', 'pc7']

components

print(model['pca'].explained_variance_ratio_)

var1 = np.cumsum(model['pca'].explained_variance_ratio_)

print(var1)

# Variance plot for PCA components obtained 
plt.plot(var1, color = "red")

# KneeLocator
# Refer the link to understand the parameters used: https://kneed.readthedocs.io/en/stable/parameters.html     

# from kneed import KneeLocator
kl = KneeLocator(range(len(var1)), var1, curve = 'convex', direction = "increasing") 
# The line is pretty linear hence Kneelocator is not able to detect the knee/elbow appropriately
kl.elbow
# plt.style.use("seaborn")
plt.plot(range(len(var1)), var1)
plt.xticks(range(len(var1)))
plt.ylabel("variance")
plt.axvline(x = kl.elbow, color = 'r', label = 'axvline - full height', ls = '--')
plt.show()

# The line is pretty linear hence Kneelocator is not able to detect the knee/elbow appropriately
# PCA for Feature Extraction

plt.plot(range(len(var1)), var1)
plt.xticks(range(len(var1)))
plt.ylabel("variance")
plt.axvline(x = 3, color = 'r', label = 'axvline - full height', ls = '--')
plt.show()

# Final dataset with manageable number of columns (Feature Extraction)

final = pd.concat([df.Customer, pca_res.iloc[:, 0:4]], axis = 1)
final.columns = ['Customer', 'pc0', 'pc1', 'pc2', 'pc3']
final

# Scatter diagram
ax = final.plot(x = 'pc0', y = 'pc1', kind = 'scatter', figsize = (12, 8))
final[['pc0', 'pc1', 'Customer']].apply(lambda x: ax.text(*x), axis = 1)


