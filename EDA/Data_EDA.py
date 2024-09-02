2 + 2 # Function F9
# Works as calculator

# Python Libraries (Packages)
# pip install <package name> - To install library (package), execute the code in Command prompt
# pip install pandas

import pandas as pd

dir(pd)

# Read data into Python
education = pd.read_csv(r"D:\Data\education.csv")
Education = pd.read_csv("D:/Data/education.csv")

A = 10
a = 10.1

education.info()

# C:\Users\education.csv - this is windows default file path with a '\'
# C:\\Users\\education.csv - change it to '\\' to make it work in Python

# Exploratory Data Analysis
# Measures of Central Tendency / First moment business decision
education.workex.mean() # '.' is used to refer to the variables within object
education.workex.median()
education.workex.mode()


# Measures of Dispersion / Second moment business decision
education.workex.var() # variance
education.workex.std() # standard deviation
range = max(education.workex) - min(education.workex) # range
range


# Third moment business decision
education.workex.skew()
education.gmat.skew()


# Fourth moment business decision
education.workex.kurt()

