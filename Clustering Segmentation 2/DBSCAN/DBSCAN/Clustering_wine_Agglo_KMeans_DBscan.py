# Problem Statement

'''
# Identifying the best quality wine is a special skill and very few experts are specialized in accurately detecting the quality.
# The objective of this project is to simplify the process of detecting the quality of wine.

# `CRISP-ML(Q)` process model describes six phases:
# 1. Business and Data Understanding
# 2. Data Preparation
# 3. Model Building
# 4. Evaluation
# 5. Model Deployment
# 6. Monitoring and Maintenance

# **Objective(s):** Minimize Shipment Organization Time
# 
# **Constraints:** Minimize Specialists' Dependency    

# **Success Criteria**
# - **Business Success Criteria**: Reduce the time of wine quality check by anywhere between 20% to 40%
# - **ML Success Criteria**: Achieve Silhouette coefficient of atleast 0.5
# - **Economic Success Criteria**: Wine distillers will see an increase in revenues by atleast 20%

# **Proposed Plan:**
# Grouping the available wines will allow to understand the characteristics of each group.

# ### Data Dictionary

# - OD_read: Amount of dilution in that particular wine type
# - Proline: Amount of Proline in that particular wine type 
# Proline is typically the most abundant amino acid present in grape juice and wine
'''

import pandas as pd # data manipulation
import sweetviz # autoEDA
import matplotlib.pyplot as plt # data visualization

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN # machine learning algorithms
from sklearn.metrics import silhouette_score

from sqlalchemy import create_engine, text # connect to SQL database

# Load Wine data set
df = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Clustering Segmentation 2/DBSCAN/DBSCAN/wine_data.csv")

# Credentials to connect to Database
user = 'root'  # user name
pw = '1234'  # password
db = 'wine_db'  # database name
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# to_sql() - function to push the dataframe onto a SQL table.
df.to_sql('wine_tbl', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

sql = text('select * from wine_tbl;')
wine_df = pd.read_sql_query(sql, engine.connect())

# Show sample of data set
wine_df.head()

# ## EXPLORATORY DATA ANALYSIS (EDA) / DESCRIPTIVE STATISTICS

# ***Descriptive Statistics and Data Distribution Function***
wine_df.describe()

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
# wine_df.skew()

# ***4th Business Moment Decision (Kurtosis)***
# Measure of peakedness - represents the overall spread in the data
# wine_df.kurt()


# AutoEDA
# ## Automated Libraries
# import sweetviz
my_report = sweetviz.analyze([wine_df, "wine_df"])

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
wine_df.info()


# **Handling duplicates:**
# If the dataset has multiple entries of the same record then we can remove the duplicate entries. In case of duplicates we will use function drop_duplicates()
duplicate = wine_df.duplicated()  # Returns Boolean Series denoting duplicate rows.
print(duplicate)

sum(duplicate)
print(wine_df.shape)

# Removing Duplicates
wine_df = wine_df.drop_duplicates() # Returns DataFrame with duplicate rows removed.
print(wine_df.shape)


# **Missing Value Analysis**

# ***IMPUTATION:***
# The process of dealing with missing values is called Imputation.
# Most popular substitution based Imputation techniques are:
# 1) Mean imputation for numeric data
# 2) Mode imputation for non-numeric data

wine_df.isnull().sum() # Check for missing values


# ### Outliers Analysis:
# Exceptional data values in a variable can be outliers. In case of outliers we can use one of the strategies of 3 R (Rectify, Retain, or Remove)

# **Box Plot**
# Visualize numeric data using boxplot for outliers

# Multiple boxplots in a single visualization.
# Columns with larger scales affect other columns. 
# Below code ensures each column gets its own y-axis.

# pandas plot() function with parameters kind = 'box' and subplots = True
wine_df.plot(kind = 'box', subplots = True, sharey = False, figsize = (10, 6)) 
'''sharey True or 'all': x- or y-axis will be shared among all subplots.
False or 'none': each subplot x- or y-axis will be independent.'''
# increase spacing between subplots
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()  
# **No outliers observed**

# ## Scatter Plot
plt.scatter(wine_df.values[:, 0], wine_df.values[:, 1])
plt.title("Wine Dataset")
plt.xlabel("OD Reading")
plt.ylabel("Proline")
plt.show()

# Correlation Coefficient
wine_df.corr()


# Generate clusters using Agglomerative Hierarchical Clustering
ac = AgglomerativeClustering(5, linkage = 'average')
ac_clusters = ac.fit_predict(wine_df)

# Generate clusters from K-Means
km = KMeans(5)
km_clusters = km.fit_predict(wine_df)

# Generate clusters using DBSCAN
db_param_options = [[20, 5], [25, 5], [30, 5], [25, 7], [35, 7], [40, 5]]

for ep, min_sample in db_param_options:
    db = DBSCAN(eps = ep, min_samples = min_sample)
    db_clusters = db.fit_predict(wine_df)
    print("Eps: ", ep, "Min Samples: ", min_sample)
    print("DBSCAN Clustering: ", silhouette_score(wine_df, db_clusters))

# Generate clusters using DBSCAN
db = DBSCAN(eps = 40, min_samples = 5)
db_clusters = db.fit_predict(wine_df)

plt.figure(1)
plt.title("Wine Clusters from Agglomerative Clustering")
plt.scatter(wine_df['OD_read'], wine_df['Proline'], c = ac_clusters, s = 50, cmap = 'tab20b')
plt.show()

plt.figure(2)
plt.title("Wine Clusters from K-Means")
plt.scatter(wine_df['OD_read'], wine_df['Proline'], c = km_clusters, s = 50, cmap = 'tab20b')
plt.show()

plt.figure(3)
plt.title("Wine Clusters from DBSCAN")
plt.scatter(wine_df['OD_read'], wine_df['Proline'], c = db_clusters, s = 50, cmap = 'tab20b')
plt.show()


# Calculate Silhouette Scores
print("Silhouette Scores for Wine Dataset:\n")

print("Agg Clustering: ", silhouette_score(wine_df, ac_clusters))

print("K-Means Clustering: ", silhouette_score(wine_df, km_clusters))

print("DBSCAN Clustering: ", silhouette_score(wine_df, db_clusters))

## saving dbscan
import pickle
pickle.dump(db, open('db.pkl', 'wb'))

model = pickle.load(open('db.pkl', 'rb'))

res = model.fit_predict(wine_df)
