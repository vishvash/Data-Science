# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 17:41:13 2024

@author: Lenovo
"""

# Load the Data
import pandas as pd

df = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/EDA/InClass_DataPreprocessing_datasets/Boston.csv")
pd.set_option('display.max_columns', None)

# Auto EDA
# ---------
# Sweetviz
# Autoviz
# Dtale
# Pandas Profiling
# Dataprep


# Sweetviz
###########
#pip install sweetviz
import sweetviz as sv

s = sv.analyze(df)
s.show_html()

df.value_counts()
df.nunique()
df.info()
print(df.describe())
df.head(5)
df.corr()


# pip install autoviz
from autoviz.AutoViz_Class import AutoViz_Class

av = AutoViz_Class()
a = av.AutoViz(r"C:/Users/Lenovo/Downloads/Study material/EDA/InClass_DataPreprocessing_datasets/Boston.csv", chart_format = 'html')

import os
os.getcwd()

# If the dependent variable is known:
a = av.AutoViz(r"C:/Users/Lenovo/Downloads/Study material/EDA/InClass_DataPreprocessing_datasets/Boston.csv", depVar = 'medv', chart_format = 'html') # depVar - target variable in your dataset

import pandas as pd

df = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/EDA/InClass_DataPreprocessing_datasets/Boston.csv")

import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in df.items():
    sns.boxplot(y=k, data=df, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
print(df.describe())

'''From get-go, two data coulmns show interesting summeries. 
They are : ZN (proportion of residential land zoned for lots over 25,000 sq.ft.) 
with 0 for 25th, 50th percentiles. Second, CHAS: Charles River dummy variable 
(1 if tract bounds river; 0 otherwise) with 0 for 25th, 50th and 75th percentiles. 
These summeries are understandable as both variables are conditional + categorical variables. 
First assumption would be that these coulms may not be useful in regression task such as 
predicting MEDV (Median value of owner-occupied homes).
'''
'''
Another interesing fact on the dataset is the max value of MEDV. 
From the original data description, it says: Variable #14 seems to be censored at 50.00 
(corresponding to a median price of $50,000). Based on that, values above 50.00 may not help to predict MEDV. 
Let's plot the dataset and see interesting trends/stats.
'''
columns_to_drop = ['chas', 'zn']
df.drop(columns_to_drop, axis=1, inplace=True)
df = df[~(df['medv'] >= 50.0)]

fig, axs = plt.subplots(ncols=6, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in df.items():
    sns.boxplot(y=k, data=df, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


# IQR = df['rm'].quantile(0.75) - df['rm'].quantile(0.25)
# lower_limit = df['rm'].quantile(0.25) - (IQR * 1.5)
# upper_limit = df['rm'].quantile(0.75) + (IQR * 1.5)
# df['rm'] = df['rm'].clip(lower=lower_limit, upper=upper_limit)
# sns.boxplot(df['rm'])
# df

fig, axs = plt.subplots(ncols=6, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in df.items():
    IQR = df[k].quantile(0.75) - df[k].quantile(0.25)
    lower_limit = df[k].quantile(0.25) - (IQR * 1.5)
    upper_limit = df[k].quantile(0.75) + (IQR * 1.5)
    df[k] = df[k].clip(lower=lower_limit, upper=upper_limit)
    sns.boxplot(y=k, data=df, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


from feature_engine.outliers import Winsorizer
winsor_iqr = Winsorizer(capping_method = 'iqr', 
                        # choose  IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 1, 
                          variables = ['black'])
df1 = winsor_iqr.fit_transform(df[['black']])
sns.boxplot(df1.black)
#iqr works for removing outliers in column "black"


from feature_engine.outliers import Winsorizer
winsor_iqr = Winsorizer(capping_method = 'quantiles', 
                        # choose  IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 0.16, 
                          variables = ['black'])
df1 = winsor_iqr.fit_transform(df[['black']])
sns.boxplot(df1.black)
# quantiles work for removing outliers in column "black"


from feature_engine.outliers import Winsorizer
winsor_iqr = Winsorizer(capping_method = 'gaussian', 
                        # choose  IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 2, 
                          variables = ['black'])
df1 = winsor_iqr.fit_transform(df[['black']])
sns.boxplot(df1.black)
# gaussian also worls for removing outliers in column "black"



fig, axs = plt.subplots(ncols=6, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in df.items():
    sns.boxplot(y=k, data=df, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)

df.info()

#==========================================================================================================

import pandas as pd
df = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/EDA/InClass_DataPreprocessing_datasets/Animal_category.csv")

### Identify duplicate records in the data ###
# Duplicates in rows
# help(df.duplicated)
# duplicate = df.duplicated()  # Returns Boolean Series denoting duplicate rows.
# duplicate
# sum(duplicate)

# Create dummy variables
df_new = pd.get_dummies(df)

df_new.columns
df_new.info()

df_new_1 = pd.get_dummies(df, drop_first = True)



from sklearn.preprocessing import OneHotEncoder
# Creating instance of One-Hot Encoder
enc = OneHotEncoder() # initializing method

enc_df = pd.DataFrame(enc.fit_transform(df.iloc[:, 1:]).toarray())

# Get the feature names from the OneHotEncoder
feature_names = enc.get_feature_names_out(input_features=df.columns[1:])

# Assign feature names to the encoded DataFrame
enc_df.columns = feature_names
enc_df.columns
enc_df.info()



import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Define the label encoder
encoder = LabelEncoder()

# Fit label encoder and transform the "Types" column
df['Encoded_Types'] = encoder.fit_transform(df['Types']) + 1  # Adding 1 to start encoding from 1
df.head()


data = pd.DataFrame({
    'Types': ['A', 'B', 'C', 'C', 'A', 'B', 'D', 'E']
})
# Define the mapping of values to alphabets
value_mapping = {'A': 4, 'B': 2, 'C': 8, 'D': 6, 'E': 10}  # Add more alphabets if needed

# Map values to the "Types" column
data['Encoded_Types2'] = data['Types'].map(value_mapping)

print(data)

#==========================================================================================================

import pandas as pd
df = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/EDA/InClass_DataPreprocessing_datasets/iris.csv")

df.info()
df_num = df.iloc[:,1:5]
df_num.corr()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_binning = df.copy()

# Fixed width binning
fixed_bins = [0, 4.5, 6.5, 8.5]  # Define bins manually
df_binning['Sepal.Length'] = pd.cut(df['Sepal.Length'], bins=fixed_bins, labels=['Small', 'Medium', 'Large'])
df_binning['Petal.Length'] = pd.cut(df['Petal.Length'], bins=fixed_bins, labels=['Small', 'Medium', 'Large'])
# Adaptive width binning
df_binning['Sepal.Width'] = pd.qcut(df['Sepal.Width'], q=2, labels=['Small', 'Large'])
df_binning['Petal.Width'] = pd.qcut(df['Petal.Width'], q=2, labels=['Small', 'Large'])

df_binning.head()

# Plot countplot for 'Sepal.Length'
plt.figure(figsize=(8, 6))
sns.countplot(data=df_binning, x='Sepal.Length', hue='Species')
plt.title('Discretized Bins for Sepal Length')
plt.xlabel('Bins')
plt.ylabel('Count')
plt.show()


# Plot countplot for 'Petal.Length'
plt.figure(figsize=(8, 6))
sns.countplot(data=df_binning, x='Petal.Length', hue='Species')
plt.title('Discretized Bins for Petal Length')
plt.xlabel('Bins')
plt.ylabel('Count')
plt.show()

# Plot countplot for 'Sepal.Width'
plt.figure(figsize=(8, 6))
sns.countplot(data=df_binning, x='Sepal.Width', hue='Species')
plt.title('Discretized Bins for Sepal Width')
plt.xlabel('Bins')
plt.ylabel('Count')
plt.show()

# Plot countplot for 'Petal.Width'
plt.figure(figsize=(8, 6))
sns.countplot(data=df_binning, x='Petal.Width', hue='Species')
plt.title('Discretized Bins for Petal Width')
plt.xlabel('Bins')
plt.ylabel('Count')
plt.show()

#==========================================================================================================


import numpy as np
import pandas as pd

# Load modified ethnic dataset
df = pd.read_csv(r'C:/Users/Lenovo/Downloads/Study material/EDA/InClass_DataPreprocessing_datasets/claimants.csv') # for doing modifications

# Check for count of NA's in each column
print(df.isna().sum())


# For Mean, Median, Mode imputation we can use Simple Imputer or df.fillna()
from sklearn.impute import SimpleImputer

import seaborn as sns
df.info()
print(df.shape)

# casewise deletion with respect to the column 'CLMSEX'
df.dropna(subset=['CLMSEX'], inplace=True)
df.CLMSEX.isna().sum()
df.reset_index(drop=True, inplace=True) #mandatory step inorder not to lose data in further imputation steps
print(df.shape)

sns.boxplot(df.CLMAGE)


# Median Imputer
median_imputer = SimpleImputer(missing_values = np.nan, strategy = 'median')
df["CLMAGE"] = pd.DataFrame(median_imputer.fit_transform(df[["CLMAGE"]]))
df["CLMAGE"].isna().sum() #using median impuation to not get affected by outlier

# Mode Imputer since there are nominal data
mode_imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
# df = pd.DataFrame(mode_imputer.fit_transform(df))
df["CLMINSUR"] = pd.DataFrame(mode_imputer.fit_transform(df[["CLMINSUR"]]))
df["SEATBELT"] = pd.DataFrame(mode_imputer.fit_transform(df[["SEATBELT"]]))
df.isnull().sum()  

mode_imputed_df = df.fillna(df.mode().iloc[0])
df.CASENUM.value_counts()

median_imputed_SEATBELT = df['SEATBELT'].fillna(df['SEATBELT'].median())
median_imputed_SEATBELT.isna().sum()


# Random Imputer
from feature_engine.imputation import RandomSampleImputer

random_imputer = RandomSampleImputer(['CLMAGE'])
df1 = pd.DataFrame(random_imputer.fit_transform(df[["CLMAGE"]]))
df1["CLMAGE"].isna().sum()  # all records replaced by median
df1.describe()
df1["CLMAGE"].median()

print(df.isna().sum())

#==========================================================================================================

import pandas as pd

# Read data into Python
df = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/EDA/InClass_DataPreprocessing_datasets/calories_consumed.csv")

import scipy.stats as stats
import pylab
import seaborn as sns
import matplotlib.pyplot as plt

df.info()

# Checking whether data is normally distributed
stats.probplot(df['Weight gained (grams)'], dist = "norm", plot = pylab)

stats.probplot(df['Calories Consumed'], dist = "norm", plot = pylab)

import numpy as np


# Transformation to make workex variable normal
stats.probplot(np.log(df['Weight gained (grams)']), dist = "norm", plot = pylab)


# Transform training data & save lambda value
fitted_data, fitted_lambda = stats.boxcox(df['Weight gained (grams)'])

# creating axes to draw plots
fig, ax = plt.subplots(1, 2)

# Plotting the original data (non-normal) and fitted data (normal)
sns.distplot(df['Weight gained (grams)'], hist = False, kde = True,
             kde_kws = {'shade': True, 'linewidth': 2},
             label = "Non-Normal", color = "green", ax = ax[0])

sns.distplot(fitted_data, hist = True, kde = True,
             kde_kws = {'shade': True, 'linewidth': 2},
             label = "Normal", color = "green", ax = ax[1])

# adding legends to the subplots
plt.legend(loc = "upper right")

# rescaling the subplots
fig.set_figheight(5)
fig.set_figwidth(10)

print(f"Lambda value used for Transformation: {fitted_lambda}")

# Transformed data
prob = stats.probplot(fitted_data, dist = stats.norm, plot = pylab)

# Yeo-Johnson Transform

# Original data
prob = stats.probplot(df['Calories Consumed'], dist = stats.norm, plot = pylab)

from feature_engine import transformation

# Set up the variable transformer
tf = transformation.YeoJohnsonTransformer(variables = 'Calories Consumed')

df = tf.fit_transform(df)

# Transformed data
prob = stats.probplot(df['Calories Consumed'], dist = stats.norm, plot = pylab)

#==========================================================================================================

import pandas as pd

df = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/EDA/InClass_DataPreprocessing_datasets/Z_dataset.csv")

df.dtypes
# If the variance is low or close to zero, then a feature is approximately constant and will not improve the performance of the model.
# In that case, it should be removed. 

ds = df.iloc[:,1:5]
ds
ds.columns

for column in ds.columns:
    range = max(ds[column]) - min(ds[column])
    print(range)

for a in ds.columns:
    print(f"the variance of {a} is ", df[a].var())
    
#since variance of square.breadth is near zero we will drop the column
df.drop(columns=["square.breadth"], inplace=True)

df['square.length'].var()
#colour has equal number of categories so we will retain it

print(df)

df.colour.value_counts()

#==========================================================================================================

import pandas as pd
import numpy as np
import seaborn as sns

data = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/EDA/InClass_DataPreprocessing_datasets/Seeds_data.csv")

a = data.describe()
print(a)

### Standardization
from sklearn.preprocessing import StandardScaler

# Initialise the Scaler
scaler = StandardScaler()

# To scale data
df = scaler.fit_transform(data)
# Convert the array back to a dataframe
dataset = pd.DataFrame(df, columns=data.columns)
res = dataset.describe()
print(res)

sns.boxplot(dataset.Assymetry_coeff)

'''Robust Scaling- robust to outliers'''

from sklearn.preprocessing import RobustScaler

robust_model = RobustScaler()

df_robust = robust_model.fit_transform(data)

dataset_robust = pd.DataFrame(df_robust, columns=data.columns)
res_robust = dataset_robust.describe()

print(res_robust)


# Normalization

from sklearn.preprocessing import MinMaxScaler
minmaxscale = MinMaxScaler()

df_n = minmaxscale.fit_transform(df)
dataset1 = pd.DataFrame(df_n)

res1 = dataset1.describe()
print(res1)

#==========================================================================================================

# Create the string
my_string = "stay positive and optimistic"

# Split the string on whitespace
split_string = my_string.split()

print("Split string:", split_string)

# a) Check if the string starts with "H"
starts_with_H = my_string.startswith("H")
print("Does the string start with 'H'?", starts_with_H)

# b) Check if the string ends with "d"
ends_with_d = my_string.endswith("d")
print("Does the string end with 'd'?", ends_with_d)

# c) Check if the string ends with "c"
ends_with_c = my_string.endswith("c")
print("Does the string end with 'c'?", ends_with_c)

# Print "🪐" one hundred and eight times
print("🪐" * 108)

# Original reversed story
reversed_story = ".elgnujehtotniffo deps mehtfohtoB .eerfnoilehttesotseporeht no dewangdnanar eh ,ylkciuQ .elbuortninoilehtdecitondnatsapdeklawesuomeht ,nooS .repmihwotdetratsdnatuotegotgnilggurts saw noilehT .eert a tsniagapumihdeityehT .mehthtiwnoilehtkootdnatserofehtotniemacsretnuhwef a ,yad enO .ogmihteldnaecnedifnocs’esuomeht ta dehgualnoilehT ”.emevasuoy fi yademosuoyotplehtaergfo eb lliw I ,uoyesimorp I“ .eerfmihtesotnoilehtdetseuqeryletarepsedesuomehtnehwesuomehttaeottuoba saw eH .yrgnaetiuqpuekow eh dna ,peels s’noilehtdebrutsidsihT .nufroftsujydobsihnwoddnapugninnurdetratsesuom a nehwelgnujehtnignipeelsecno saw noil A"

# Split the story into sentences
sentences = reversed_story.split(" .")

# Reverse the order of the sentences
reversed_sentences = sentences[::-1]

# Join the reversed sentences and reverse each sentence's characters
correct_order_story = ""
for sentence in reversed_sentences:
    correct_order_story += sentence[::-1] + ". "

# Print the correct order story
print(correct_order_story)

#==========================================================================================================

import pandas as pd

data = pd.read_excel(r"C:/Users/Lenovo/Downloads/Study material/EDA/InClass_DataPreprocessing_datasets/Online Retail1.xlsx")
data.dtypes

'''
CustomerID is float- Python automatically identify the data types by interpreting the values. 
As the data for CustomerID is numeric Python detects the values as float64.

From measurement levels prespective the CustomerID is a Nominal data as it is an identity for each employee.

If we have to alter the data type which is defined by Python then we can use astype() function

'''


# Convert 'int64' to 'str' (string) type. 
data.CustomerID = data.CustomerID.astype('str')
data['CustomerID']
data.dtypes


### Identify duplicate records in the data ###

# Duplicates in rows

duplicate = data.duplicated()  # Returns Boolean Series denoting duplicate rows.
duplicate

sum(duplicate)
pd.set_option('display.max_columns', None)

# Removing Duplicates
data = data.drop_duplicates() # Returns DataFrame with duplicate rows removed.

# Parameters
duplicate = data.duplicated()
sum(duplicate)

# Duplicates in Columns
# We can use correlation coefficient values to identify columns which have duplicate information


# Correlation coefficient
'''
Ranges from -1 to +1. 
Rule of thumb says |r| > 0.85 is a strong relation
'''
data[['UnitPrice','Quantity']].corr()

import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(data)
sns.histplot(data)

sns.boxplot(data=data, x='Country', y='UnitPrice')
plt.title('Box Plot of UnitPrice by Country')
plt.xticks(rotation=45)
plt.show()

sns.histplot(data=data, x='Quantity', bins=10, kde=True)
plt.title('Histogram of Quantity')
plt.show()

sns.countplot(data=data, x='Country')
plt.title('Count of Transactions by Country')
plt.xticks(rotation=45)
plt.show()

sns.scatterplot(data=data, x='Quantity', y='UnitPrice')
plt.title('Scatter Plot of Quantity vs. UnitPrice')
plt.show()


