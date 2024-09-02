# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 14:18:01 2024

@author: Lenovo
"""
"""
1)	Connect MySQL using Python, push "Copy of Data.csv" into the database.
"""


import pandas as pd
# Import data (.csv file) using pandas.
data = pd.read_csv("Downloads//Copy of Data.csv")

# pip install pymysql

# pip install sqlalchemy
from sqlalchemy import create_engine
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user = "root",# user
                               pw = "1234", # passwrd
                               db = "dataanalytics_db")) #database

data.to_sql('data', con = engine, if_exists = 'replace', chunksize = None, index= False) # sending data into database and connecting with Engine by using "DataFrame.to_sql()"


"""
2)	Connect MySQL using Python, push "Copy of Indian Cities.csv" into the database, and then use Python to load the data from MySQL.
"""

cities = pd.read_csv("Downloads//Copy of Indian_cities.csv")

# pip install pymysql

# # pip install sqlalchemy
# from sqlalchemy import create_engine
# engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
#                        .format(user = "root",# user
#                                pw = "1234", # passwrd
#                                db = "dataanalytics_db")) #database

cities.to_sql('cities', con = engine, if_exists = 'replace', chunksize = None, index= False) # sending data into database and connecting with Engine by using "DataFrame.to_sql()"

# To get the data From DataBase
sql = "SELECT * FROM cities;" # wright query of sql and save into variable
loaded_cities = pd.read_sql_query(sql, engine) # connecting query with Engine and reading the results by using "pd.read_sql_query"
