# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 19:56:19 2024

@author: Lenovo
"""

"""
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

Objective: Maximize the operational efficiency
Constraints: Maximize the financial health



Success Criteria: 
Business Success Criteria: Increase the operational efficiency by 10% to 12% by segmenting the Airlines.
ML Success Criteria: Achieve a Silhouette coefficient of at least 0.7
Economic Success Criteria: The airline companies will see an increase in revenues by at least 8% (hypothetical numbers)

"""

pip install py-AutoClean

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sweetviz
from AutoClean import AutoClean

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline

from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering

from sqlalchemy import create_engine, text

traff = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Hierarchical Clustering_Hands-on/Assignments/Data Set/Data Set (5)/AirTraffic_Passenger_Statistics.csv")

user = "root"
pw = 1234
db = "traff_db"
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")


# to_sql() - function to push the dataframe onto a SQL table.
traff.to_sql('traff_tbl', con = engine, if_exists = 'replace', chunksize = 1000, index = False)



###### To read the data from MySQL Database
sql = 'select * from traff_tbl;'


df = pd.read_sql_query(text(sql), engine.connect())


# Data types
df.info()

# EXPLORATORY DATA ANALYSIS (EDA) / DESCRIPTIVE STATISTICS
# ***Descriptive Statistics and Data Distribution Function***

df.describe()


# Data Preprocessing

# **Cleaning Unwanted columns**
# UnivID is the identity to each university. 
# Analytically it does not have any value (Nominal data). 
# We can safely ignore the ID column by dropping the column.

df.drop(['Operating Airline IATA Code', 'Boarding Area', 'Year', 'Month'], axis = 1, inplace = True)

df.info()

# ## Automated Libraries

# AutoEDA
# import sweetviz
my_report = sweetviz.analyze([df, "df"])

my_report.show_html('Report.html')


'''
Alternatively, we can use other AutoEDA functions as well.
# D-Tale
########

pip install dtale
import dtale

d = dtale.show(df)
d.open_browser()

'''

# EDA report highlights:
# ------------------------
# Missing Data: Identified Missing Data in columns: SAT, GradRate

# Outliers:  Detected exceptional values in 4 columns: SAT, Top10, Accept, SFRatio
# Boxplot

#Install PyQt5 if you get this warning message - "UserWarning:Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure."
#pip install PyQt5
#import PyQt5

df.plot(kind = 'box', subplots = True, sharey = False, figsize = (15, 8)) 

# increase spacing between subplots
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()

# Encoding: 'State' is categorical data that needs to be encoded into numeric values


# Data Preprocessing
# -----------------------------------------------------------------------------
# Auto Preprocessing and Cleaning
# from AutoClean import AutoClean
clean_pipeline = AutoClean(df.iloc[:, :], mode = 'manual', missing_num = 'auto'
                           #, outliers = 'winz')
                           , encode_categ = ['auto'])


# help(AutoClean)

# Missing values = 'auto': AutoClean first attempts to predict the missing values with Linear Regression
# outliers = 'winz': outliers are handled using winsorization
# encode_categ = 'auto': Label encoding performed (if more than 10 categories are present)

df_clean = clean_pipeline.output
df_clean.head()
df_clean.info()

numeric_features = df_clean.select_dtypes(exclude = ['object']).columns
numeric_features

categorical_features = df_clean.select_dtypes(include = ['object']).columns
categorical_features

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


num_pipeline = Pipeline([('impute', SimpleImputer(strategy = 'mean')), ('scale', MinMaxScaler())])
num_pipeline

categ_pipeline = Pipeline([('OnehotEncode', OneHotEncoder(drop = 'first'))])
categ_pipeline

# Using ColumnTransfer to transform the columns of an array or pandas DataFrame. 
# This estimator allows different columns or column subsets of the input to be
# transformed separately and the features generated by each transformer will
# be concatenated to form a single feature space.
preprocess_pipeline = ColumnTransformer([('categorical', categ_pipeline, categorical_features), 
                                       ('numerical', num_pipeline, numeric_features)], 
                                        remainder = 'passthrough') # Skips the transformations for remaining columns

preprocess_pipeline

processed2 = preprocess_pipeline.fit(df_clean) 

df_clean = pd.DataFrame(processed2.transform(df_clean).toarray(), columns = list(processed2.get_feature_names_out()))


# #### Drawback with this approach: If there are more than 10 categories, then Autoclean performs label encoding.

# df_clean.drop(['State'], axis = 1, inplace = True)

df_clean.head()

# -----------------------------------------------------------------------------


# ## Normalization/MinMax Scaler - To address the scale differences

# ### Python Pipelines
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import MinMaxScaler

df_clean.info()

cols = list(df_clean.columns)
print(cols)

pipe1 = make_pipeline(MinMaxScaler())


cols_n = list(df_clean.columns)


# Train the data preprocessing pipeline on data
df_pipelined = pd.DataFrame(pipe1.fit_transform(df_clean), columns = cols_n, index = df_clean.index)

df_pipelined.head()



df_pipelined.describe() # scale is normalized to min = 0; max = 1

###### End of Data Preprocessing ######
# -----------------------------------------------------------------------------


######### Model Building #########
# # CLUSTERING MODEL BUILDING

# ### Hierarchical Clustering - Agglomerative Clustering

# from scipy.cluster.hierarchy import linkage, dendrogram
# from sklearn.cluster import AgglomerativeClustering 
# import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline') --- if running in jupyter notebook

plt.figure(1, figsize = (16, 8))
tree_plot = dendrogram(linkage(df_pipelined, method  = "complete"))

plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Euclidean distance')
plt.show()


# Applying AgglomerativeClustering and grouping data into 3 clusters 
# based on the above dendrogram as a reference
hc1 = AgglomerativeClustering(n_clusters = 14, 
                              #affinity = 'euclidean', 
                              linkage = 'ward')
#===========================rough practice=====================================
# from sklearn.metrics.pairwise import pairwise_distances


# dissimilarity_matrix = pairwise_distances(df, metric='euclidean')


# # Perform agglomerative clustering with Ward linkage using Gower dissimilarity
# hc1 = AgglomerativeClustering(n_clusters=14, linkage='ward', affinity='precomputed')
# clusters = hc1.fit_predict(dissimilarity_matrix)


# pip install gower
# from gower import gower_matrix

# gower_dist = gower_matrix(df)

# hc1 = AgglomerativeClustering(n_clusters=14, linkage='ward')

# help(AgglomerativeClustering())


# z_hc1 = hc1.fit_predict(gower_dist)
# hc1.labels_

# cluster_labels = pd.Series(hc1.labels_) 

# df_clust = pd.concat([cluster_labels, df], axis = 1) 

# df_clust.head()

# df_clust.columns
# df_clust = df_clust.rename(columns = {0: 'cluster'})
# df_clust.head()


# metrics.silhouette_score(gower_dist, cluster_labels)


#==============================================================================

y_hc1 = hc1.fit_predict(df_pipelined)
y_hc1

# Analyzing the Results obtained
hc1.labels_   # Referring to the cluster labels assigned

cluster_labels = pd.Series(hc1.labels_) 

# Combine the labels obtained with the data
df_clust = pd.concat([cluster_labels, df_clean], axis = 1) 

df_clust.head()

df_clust.columns
df_clust = df_clust.rename(columns = {0: 'cluster'})
df_clust.head()



# # Clusters Evaluation

# **Silhouette coefficient:**  
# Silhouette coefficient is a Metric, which is used for calculating 
# goodness of the clustering technique, and the value ranges between (-1 to +1).
# It tells how similar an object is to its own cluster (cohesion) compared to 
# other clusters (separation).
# A score of 1 denotes the best meaning that the data point is very compact 
# within the cluster to which it belongs and far away from the other clusters.
# Values near 0 denote overlapping clusters.

# from sklearn import metrics
metrics.silhouette_score(df_pipelined, cluster_labels)

'''Alternatively, we can use:'''
# **Calinski Harabasz:**
# Higher value of the CH index means clusters are well separated.
# There is no thumb rule which is an acceptable cut-off value.
metrics.calinski_harabasz_score(df_pipelined, cluster_labels)

# **Davies-Bouldin Index:**
# Unlike the previous two metrics, this score measures the similarity of clusters. 
# The lower the score the better the separation between your clusters. 
# Vales can range from zero and infinity
metrics.davies_bouldin_score(df_pipelined, cluster_labels)



'''Hyperparameter Optimization for Hierarchical Clustering'''
# Experiment to obtain the best clusters by altering the parameters

# ## Cluster Evaluation Library

# pip install clusteval
# Refer to link: https://pypi.org/project/clusteval

# from clusteval import clusteval
# import numpy as np

# Silhouette cluster evaluation. 
ce = clusteval(evaluate = 'silhouette')

df_array = np.array(df_pipelined)

# Fit
ce.fit(df_array)

# Plot
ce.plot()

## Using the report from clusteval library building 2 clusters
# Fit using agglomerativeClustering with metrics: euclidean, and linkage: ward

hc_2clust = AgglomerativeClustering(n_clusters = 15, 
                                    #affinity = 'euclidean', 
                                    linkage = 'ward')

y_hc_2clust = hc_2clust.fit_predict(df_pipelined)

# Cluster labels
hc_2clust.labels_

cluster_labels2 = pd.Series(hc_2clust.labels_) 

# Concate the Results with data
df_2clust = pd.concat([cluster_labels2, df_clean], axis = 1)

df_2clust = df_2clust.rename(columns = {0:'cluster'})
df_2clust.head()

pd.set_option('display.max_columns', None)

# Aggregate using the mean of each cluster
df_2clust.iloc[:, 1:].groupby(df_2clust.cluster).mean()

# Save the Results to a CSV file
df_3clust = pd.concat([df, cluster_labels2], axis = 1)

df_3clust = df_3clust.rename(columns = {0:'cluster'})

df_3clust.select_dtypes(include=['float64', 'int64']).groupby('cluster').mean()

df_3clust.dtypes

df_3clust.to_csv('Airtraffic.csv', encoding = 'utf-8')

import os
os.getcwd()


