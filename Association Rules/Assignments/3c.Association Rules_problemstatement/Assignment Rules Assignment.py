# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 22:00:46 2024

@author: Lenovo
"""

'''
# Data Mining Unsupervised Learning / Descriptive Modeling - Association Rule Mining

# Problem Statement
Kitabi Duniya, a famous bookstore in India, was established before Independence, the growth of the company was incremental year by year, but due to the online selling of books and widespread Internet access, its annual growth started to collapse. As a Data Scientist, you must help this heritage bookstore gain its popularity back and increase the footfall of customers and provide ways to improve the business exponentially to an expected value at a 25% improvement of the current rate. Apply the pattern mining techniques (Association Rules Algorithm) to identify ways to improve sales. Explain the rules (patterns) identified, and visually represent the rules in graphs for a clear understanding of the solution.

# `CRISP-ML(Q)` process model describes six phases:
# 
# 1. Business and Data Understanding
# 2. Data Preparation
# 3. Model Building
# 4. Model Evaluation
# 5. Deployment
# 6. Monitoring and Maintenance

# **Objective(s):** Maximize Sales
# 
# **Constraints:** Minimize Marketing Cost

# **Success Criteria**
# 
# - **Business Success Criteria**: Improve the sales in Retail Store
 by 15% - 20%
# 
# - **ML Success Criteria**: Accuracy : NA; 
    Performance : Complete processing within 5 mins on every quarter data
# 
# - **Economic Success Criteria**: Increase the Store profits by
 atleast 15%
# 
# **Proposed Plan:**
# Identify the Association between the books being purchased by the customers
 from the store

# ## Data Collection

# Data: 
#    The daily transactions made by the customers are captured by the store.
# 
# Description:
# A total of 2001 transactions data captured for the month.
'''

# Mlxtend (machine learning extensions) is a Python library of useful tools for
# the day-to-day data science tasks.

# pip install mlxtend


# Install the required packages if not available
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

from sqlalchemy import create_engine, text
import pickle

engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}".format(user = "root", pw = "1234", db = "retail")) # database

# Read csv file 
data = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Association Rules/Assignments/Data Set/book.csv" )
data.head()

# Load the data into MySQL DB
data.to_sql('bookshop', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

##############



# Read data from database
sql = text('select * from bookshop;')
transf_df = pd.read_sql_query(sql, con = engine.connect())

transf_df.head()

# Suppress the Warnings
import warnings
warnings.filterwarnings("ignore")

### Elementary Analysis ###
# Most popular items
count = transf_df.loc[:, :].sum()

# Generates a series
pop_item = count.sort_values(axis = 0, ascending = False).head(10)

# Convert the series into a dataframe 
pop_item = pop_item.to_frame() # type casting

# Reset Index
pop_item = pop_item.reset_index()
pop_item

pop_item = pop_item.rename(columns = {"index": "items", 0: "count"})
pop_item

# Data Visualization
# get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10, 6) # rc stands for runtime configuration 
plt.style.use('dark_background')
pop_item.plot.barh()
plt.title('Most popular items')
plt.gca().invert_yaxis() # gca means "get current axes"

help(apriori)


# Itemsets
frequent_itemsets = apriori(transf_df, min_support = 0.035, max_len = 3, use_colnames = True)
frequent_itemsets


# Most frequent itemsets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True)
frequent_itemsets

# Association Rules
rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.head(10)

rules.sort_values('lift', ascending = False).head(10)


# Handling Profusion of Rules (Duplication elimination)
def to_list(i):
    return (sorted(list(i)))

# Sort the items in Antecedents and Consequents based on Alphabetical order
ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

# Sort the merged list of items - transactions
ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

# No duplication of transactions
unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]

# Capture the index of unique item sets
index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))
    
index_rules

# Rules without any redudancy 
rules_no_redundancy = rules.iloc[index_rules, :]
rules_no_redundancy


# Sorted list and top 10 rules 
rules10 = rules_no_redundancy.sort_values('lift', ascending = False).head(10)
rules10

rules10.plot(x = "support", y = "confidence", c = rules10.lift, 
             kind = "scatter", s = 12, cmap = plt.cm.coolwarm)

rules10.info()

# Store the rules on to SQL database
# Database do not accepting frozensets

# Removing frozenset from dataframe
rules10['antecedents'] = rules10['antecedents'].astype('string')
rules10['consequents'] = rules10['consequents'].astype('string')

rules10['antecedents'] = rules10['antecedents'].str.removeprefix("frozenset({")
rules10['antecedents'] = rules10['antecedents'].str.removesuffix("})")

rules10['consequents'] = rules10['consequents'].str.removeprefix("frozenset({")
rules10['consequents'] = rules10['consequents'].str.removesuffix("})")

rules10.to_sql('bookshop_ar', con = engine, if_exists = 'replace', chunksize = 1000, index = False)