# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 12:05:53 2024

@author: Lenovo
"""

import pandas as pd

pd.set_option('display.max_columns', None)

df = pd.read_csv(r"C:/Users/Lenovo/Downloads/Employee attrition.csv")

df.dtypes
df.info()
df.describe()

#First moment of business decision
df.mean()
df.median()
df.mode()

#Second moment of business decision
df.std()
df.var()
df.max()

#Range for numerical columns

#Third moment of business decision
df.skew()

#Fourth moment of business decision
df.kurt()

data = df.loc [:, ["Age", "Attrition", "Department", "Education", "DistanceFromHome", "Gender", 
                   "JobRole", "MonthlyIncome", "PerformanceRating", "PercentSalaryHike",
                   "TotalWorkingYears", "WorkLifeBalance"] ]

data.dtypes
data.info()
data.describe()

# def ranges(dataset):
#     numerical_columns = dataset.select_dtypes(include=['number'])

#     # Calculate range for each numerical column
#     data_ranges = numerical_columns.apply(lambda x: x.max() - x.min())
#     return df_ranges 

# ranges(df)


data_numerical = data.select_dtypes(include=['number'])

# Calculate range for each numerical column
data_ranges = data_numerical.apply(lambda x: x.max() - x.min())
data_ranges 

#First moment of business decision
data_numerical.mean()
data_numerical.median()
data_numerical.mode()

#Second moment of business decision
data_numerical.std()
data_numerical.var()
data_numerical.max()

#Range for numerical columns

#Third moment of business decision
data_numerical.skew()

#Fourth moment of business decision
data_numerical.kurt()

#Input columns ['PerformanceRating'] have low variation for method 'iqr'. Try other capping methods or drop these columns.

data_numerical = data_numerical.drop("PerformanceRating", axis=1)


import seaborn as sns
import matplotlib.pyplot as plt

for i in data_numerical.columns:
    sns.boxplot(data_numerical[i])
    plt.xlabel(i)
    plt.show()
    

duplicate = data.duplicated()  # Returns Boolean Series denoting duplicate rows.
duplicate

sum(duplicate)


data_numerical.corr()
data_numerical.var()
    
from feature_engine.outliers import Winsorizer

numerical_columns = data_numerical.columns.tolist()

# Define the model with IQR method
winsor_iqr = Winsorizer(capping_method = 'iqr', 
                        # choose  IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 1.5, 
                          variables = numerical_columns)

data_numerical_prep = winsor_iqr.fit_transform(data_numerical)


for i in data_numerical_prep.columns:
    sns.boxplot(data_numerical_prep[i])
    plt.xlabel(i)
    plt.show()
    