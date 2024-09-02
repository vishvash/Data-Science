'''
# Data Mining Unsupervised Learning / Descriptive Modeling - Association Rule Mining

# Problem Statement

# Sales of the bricks and mortar stores has been less in comparison to the 
competitors in the surroundings. 
Store owner realised this by visiting various stores as part of experiments.

# Retail Store (Client) wants to leverage on the transactions data that is being captured. 
The customers purchasing habits needed to be understood by finding the 
association between the products in the customers transactions. 
This information can help Retail Store (client) to determine the shelf placement 
and by devising strategies to increase revenues and develop effective sales strategies.

# `CRISP-ML(Q)` process model describes six phases:
# 
# 1. Business and Data Understanding
# 2. Data Preparation
# 3. Model Building
# 4. Model Evaluation
# 5. Deployment
# 6. Monitoring and Maintenance

# **Objective(s):** Maximize Profits
# 
# **Constraints:** Minimize Marketing Cost

# **Success Criteria**
# 
# - **Business Success Criteria**: Improve the cross selling in Retail Store
 by 15% - 20%
# 
# - **ML Success Criteria**: Accuracy : NA; 
    Performance : Complete processing within 5 mins on every quarter data
# 
# - **Economic Success Criteria**: Increase the Retail Store profits by
 atleast 15%
# 
# **Proposed Plan:**
# Identify the Association between the products being purchased by the customers
 from the store

# ## Data Collection

# Data: 
#    The daily transactions made by the customers are captured by the store.
# 
# Description:
# A total of 9835 transactions data captured for the month.
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

list1 = ['kumar', '360DigiTMG', 2019, 2022]

print(list1)

print(len(list1))

print(list1[0])
print(list1[3])
print(list1[:3])

# Reversing list
print(list1[::-1])

del(list1[0])

print(list1)

# Create a tuple dataset
tup1 = ('kumar', '360DigiTMG', 2019, 2022)

tup2 = (1, 2, 3, 4, 5, 6, 7)

print(tup1)

print(tup2)

print(tup1[0])

print(tup2[1:5])


tup1[1]   

# We cannot delete individual items of tuple as it is immutable.
#del(tup1[1])


# Connecting to sql by creating sqlachemy engine
# user = 'user1'
# pw = 'user1'
# db = 'retail'

engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}".format(user = "root", pw = "1234", db = "retail")) # database

# Read csv file 
data = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Association Rules/Association_Rules/Association_Rules/groceries.csv", sep = ';', header = None )
data.head()

# Load the data into MySQL DB
data.to_sql('groceries', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

##############



# Read data from database
sql = text('select * from groceries;')
groceries = pd.read_sql_query(sql, con = engine.connect())

groceries.head()

# Suppress the Warnings
import warnings
warnings.filterwarnings("ignore")

# Convert the groceries into list
groceries = groceries.iloc[:, 0].to_list()
groceries

# Extract items from the transactions
groceries_list = []

for i in groceries:
   groceries_list.append(i.split(","))

print(groceries_list)


# Removing null values from list
groceries_list_new = []

for i in  groceries_list:
   groceries_list_new.append(list(filter(None, i)))

print(groceries_list_new)


# TransactionEncoder: Encoder for transaction data in Python lists
# Encodes transaction data in the form of a Python list of lists,
# into a NumPy array

TE = TransactionEncoder()
X_1hot_fit = TE.fit(groceries_list)


# import pickle
pickle.dump(X_1hot_fit, open('TE.pkl', 'wb'))

import os
os.getcwd()

X_1hot_fit1 = pickle.load(open('TE.pkl', 'rb'))

X_1hot = X_1hot_fit1.transform(groceries_list) 

print(X_1hot)


transf_df = pd.DataFrame(X_1hot, columns = X_1hot_fit1.columns_)
transf_df
transf_df.shape


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
frequent_itemsets = apriori(transf_df, min_support = 0.0075, max_len = 4, use_colnames = True)
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
    # return (list(i))
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

rules10.to_sql('groceries_ar', con = engine, if_exists = 'replace', chunksize = 1000, index = False)


