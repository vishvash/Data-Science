# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 17:05:30 2024

@author: Lenovo
"""


'''
# K-Means Clustering Algorithm - Data Mining (Machine Learning) Unsupervised learning Algorithm

Problem Statements:
Global air travel has seen an upward trend in recent times. The maintenance of operational efficiency and maximizing profitability are crucial for airlines and airport authorities. Businesses need to optimize airline and terminal operations to enhance passenger satisfaction, improve turnover rates, and increase overall revenue. 
The airline companies with the available data want to find an opportunity to analyze and understand travel patterns, customer demand, and terminal usage.

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
ML Success Criteria: Achieve a Silhouette coefficient of at least 0.6
Economic Success Criteria: The insurance company will see an increase in revenues by at least 8% 

Data: Refer to the ‘AirTraffic_Passenger_Statistics.csv’ dataset.

'''

import pandas as pd # data manipulation
import sweetviz # autoEDA
import matplotlib.pyplot as plt # data visualization

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN # machine learning algorithms
from sklearn.metrics import silhouette_score

from sqlalchemy import create_engine, text # connect to SQL database

# Load Wine data set
traff = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Clustering Segmentation 2/Assignments/Data Set/Data Set (5)/AirTraffic_Passenger_Statistics.csv")

# Credentials to connect to Database
user = 'root'  # user name
pw = '1234'  # password
db = 'traff_db'  # database name
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# to_sql() - function to push the dataframe onto a SQL table.
traff.to_sql('traff_tbl', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

###### To read the data from MySQL Database
sql = 'select * from traff_tbl;'

df = pd.read_sql_query(text(sql), engine.connect())
traff_df = df.iloc[:1000 , :]
traff_df = traff_df.select_dtypes(exclude= ['object'])
traff_df = traff_df.drop(columns ="Year")

# Show sample of data set
traff_df.head()

# ## EXPLORATORY DATA ANALYSIS (EDA) / DESCRIPTIVE STATISTICS

# ***Descriptive Statistics and Data Distribution Function***
traff_df.describe()

# ***1st Moment Business Decision (Measures of Central Tendency)***
# 1) Mean
# 2) Median
# 3) Mode

# ***2nd Moment Business Decision (Measures of Dispersion)***
# 1) Variance
# 2) Standard deviation
# 3) Range (maximum - minimum)

# ***3rd Business Moment Decision (Skewness)***
# Measure of asymmetry in the data distribution
# traff_df.skew()

# ***4th Business Moment Decision (Kurtosis)***
# Measure of peakedness - represents the overall spread in the data
# traff_df.kurt()


# AutoEDA
# ## Automated Libraries
# import sweetviz
my_report = sweetviz.analyze([traff_df, "traff_df"])

my_report.show_html('Report.html')


# ## Data Preprocessing and Cleaning

# **Typecasting** :
# 
# As Python automatically interprets the data types, there may be a requirement
# for the data type to be converted. The process of converting one data type
# to another data type is called Typecasting.
# 
# Example: 
# 1) int to float
# 2) float to int
traff_df.info()


# **Handling duplicates:**
# If the dataset has multiple entries of the same record then we can remove the duplicate entries. In case of duplicates we will use function drop_duplicates()
duplicate = traff_df.duplicated()  # Returns Boolean Series denoting duplicate rows.
print(duplicate)

sum(duplicate)
print(traff_df.shape)

# Removing Duplicates
traff_df = traff_df.drop_duplicates() # Returns DataFrame with duplicate rows removed.
print(traff_df.shape)


# **Missing Value Analysis**

# ***IMPUTATION:***
# The process of dealing with missing values is called Imputation.
# Most popular substitution based Imputation techniques are:
# 1) Mean imputation for numeric data
# 2) Mode imputation for non-numeric data

traff_df.isnull().sum() # Check for missing values


# ### Outliers Analysis:
# Exceptional data values in a variable can be outliers. In case of outliers we can use one of the strategies of 3 R (Rectify, Retain, or Remove)

# **Box Plot**
# Visualize numeric data using boxplot for outliers

# Multiple boxplots in a single visualization.
# Columns with larger scales affect other columns. 
# Below code ensures each column gets its own y-axis.

# pandas plot() function with parameters kind = 'box' and subplots = True
traff_df.plot(kind = 'box', subplots = True, sharey = False, figsize = (10, 6)) 
'''sharey True or 'all': x- or y-axis will be shared among all subplots.
False or 'none': each subplot x- or y-axis will be independent.'''
# increase spacing between subplots
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()  
# **No outliers observed**

# ## Scatter Plot
plt.scatter(traff_df.values[:, 0], traff_df.values[:, 1])
plt.title("Wine Dataset")
plt.xlabel("OD Reading")
plt.ylabel("Proline")
plt.show()

# Correlation Coefficient
traff_df.corr()


# Generate clusters using Agglomerative Hierarchical Clustering
ac = AgglomerativeClustering(5, linkage = 'average')
ac_clusters = ac.fit_predict(traff_df)

# Generate clusters from K-Means
km = KMeans(2)
km_clusters = km.fit_predict(traff_df)

# Generate clusters using DBSCAN
db_param_options = [[5000, 20], [10000, 35], [20000, 15], [15000, 25], [8000, 30], [18000, 20]]

for ep, min_sample in db_param_options:
    db = DBSCAN(eps = ep, min_samples = min_sample)
    db_clusters = db.fit_predict(traff_df)
    print("Eps: ", ep, "Min Samples: ", min_sample)
    print("DBSCAN Clustering: ", silhouette_score(traff_df, db_clusters))

# Generate clusters using DBSCAN
db = DBSCAN(eps = 2000, min_samples = 20)
db_clusters = db.fit_predict(traff_df)

plt.figure(1)
plt.title("Wine Clusters from Agglomerative Clustering")
plt.scatter(traff_df.values[:, 0], traff_df.values[:, 1], c = ac_clusters, s = 50, cmap = 'tab20b')
plt.show()

plt.figure(2)
plt.title("Wine Clusters from K-Means")
plt.scatter(traff_df.values[:, 0], traff_df.values[:, 1], c = km_clusters, s = 50, cmap = 'tab20b')
plt.show()

plt.figure(3)
plt.title("Wine Clusters from DBSCAN")
plt.scatter(traff_df.values[:, 0], traff_df.values[:, 1], c = db_clusters, s = 50, cmap = 'tab20b')
plt.show()


# Calculate Silhouette Scores
print("Silhouette Scores for Wine Dataset:\n")

print("Agg Clustering: ", silhouette_score(traff_df, ac_clusters))

print("K-Means Clustering: ", silhouette_score(traff_df, km_clusters))

print("DBSCAN Clustering: ", silhouette_score(traff_df, db_clusters))

## saving dbscan
import pickle
pickle.dump(db, open('db_assignment.pkl', 'wb'))

model = pickle.load(open('db_assignment.pkl', 'rb'))

res = model.fit_predict(traff_df)
