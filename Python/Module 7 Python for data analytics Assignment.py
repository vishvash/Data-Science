# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 13:50:54 2024

@author: Lenovo
"""

"""

1)	Please take care of missing data present in the “Data.csv” file using python module 
“sklearn.impute” and its methods, also collect all the data that has “Salary” less than “70,000”.
"""

import pandas as pd
from sklearn.impute import SimpleImputer

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv("Downloads//Data.csv")

# Step 1: Handling Missing Data using SimpleImputer
# Create a SimpleImputer object with a strategy (e.g., mean, median, etc.)
imputer = SimpleImputer(strategy='mean')

# Apply the imputer to fill missing values in numeric columns (e.g., Salary)
df[['Salaries']] = imputer.fit_transform(df[['Salaries']])

# Step 2: Collect data with "Salaries" less than 70,000
filtered_data = df[df['Salaries'] < 70000]

# Display the modified DataFrame and the filtered data
print("DataFrame after handling missing data:")
print(df)

print("\nData with 'Salary' less than 70,000:")
print(filtered_data)

# Save the modified DataFrame to a new CSV file if needed
# df.to_csv("Modified_Data.csv", index=False)

"""

2)	Subtracting dates: 
Python date objects let us treat calendar dates as something similar to numbers: we can compare them, sort them, add, and even subtract them. Do math with dates in a way that would be a pain to do by hand. The 2007 Florida hurricane season was one of the busiest on record, with 8 hurricanes in one year. The first one hit on May 9th, 2007, and the last one hit on December 13th, 2007. How many days elapsed between the first and last hurricane in 2007?
	Instructions:
	Import date from datetime.
	Create a date object for May 9th, 2007, and assign it to the start variable.
	Create a date object for December 13th, 2007, and assign it to the end variable.
	Subtract start from end, to print the number of days in the resulting timedelta object.
"""
from datetime import date

# Create date objects for May 9th, 2007, and December 13th, 2007
start = date(2007, 5, 9)
end = date(2007, 12, 13)

# Calculate the number of days elapsed
days_elapsed = end - start

# Print the result
print("Number of days elapsed:", days_elapsed.days)

"""
3)	Representing dates in different ways
Date objects in Python have a great number of ways they can be printed out as strings. In some cases, you want to know the date in a clear, language-agnostic format. In other cases, you want something which can fit into a paragraph and flow naturally.
Print out the same date, August 26, 1992 (the day that Hurricane Andrew made landfall in Florida), in a number of different ways, by using the “ .strftime() ” method. Store it in a variable called “Andrew”. 
Instructions: 	
Print it in the format 'YYYY-MM', 'YYYY-DDD' and 'MONTH (YYYY)'
"""

from datetime import date

# Create a date object for August 26, 1992
Andrew = date(1992, 8, 26)

# Print the date in different formats using strftime
print("Formatted date 'YYYY-MM':", Andrew.strftime('%Y-%m'))
print("Formatted date 'YYYY-DDD':", Andrew.strftime('%Y-%j'))
print("Formatted date 'MONTH (YYYY)':", Andrew.strftime('%B (%Y)'))

"""
4)	For the dataset “Indian_cities”, 
a)	Find out top 10 states in female-male sex ratio
b)	Find out top 10 cities in total number of graduates
c)	Find out top 10 cities and their locations in respect of  total effective_literacy_rate.
"""
import pandas as pd

# Load the dataset
indian_cities = pd.read_csv("Downloads//Indian_cities.csv")

# Task (a): Find out top 10 states in female-male sex ratio
top_states_sex_ratio = indian_cities.groupby('state_name')['sex_ratio'].mean()
top_states_sex_ratio = top_states_sex_ratio.sort_values(ascending=False).head(10)
print("Top 10 states in female-male sex ratio:")
print(top_states_sex_ratio)
print(top_states_sex_ratio.index)

# Task (b): Find out top 10 cities in total number of graduates
top_cities_graduates = indian_cities.sort_values(by='total_graduates', ascending=False).head(10)
print("Top 10 cities in total number of graduates:")
print(top_cities_graduates[['name_of_city', 'total_graduates']])
print()

# Task (c): Find out top 10 cities and their locations in respect of total effective_literacy_rate
top_cities_literacy_rate = indian_cities.sort_values(by='effective_literacy_rate_total', ascending=False).head(10)
print("Top 10 cities and their locations in respect of total effective_literacy_rate:")
print(top_cities_literacy_rate[['name_of_city', 'state_name', 'effective_literacy_rate_total']])

"""
5)	 For the data set “Indian_cities”
a)	Construct histogram on literates_total and comment about the inferences
b)	Construct scatter  plot between  male graduates and female graduates
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
indian_cities = pd.read_csv("Downloads//Indian_cities.csv")

# Task (a): Construct histogram on literates_total across the states
sorted_data = indian_cities.sort_values(by='literates_total', ascending=False)

# Plotting the bar chart
plt.figure(figsize=(12, 6))
plt.bar(sorted_data['state_name'], sorted_data['literates_total'], color='skyblue', alpha=0.7)
plt.title('Distribution of Literates Total Across Indian Cities')
plt.xlabel('States')
plt.ylabel('Total Literates')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()

"""Inference: 
The histogram provides a distribution of total literates across different cities.
You can observe the frequency of states with various levels of literate populations.
If the histogram is skewed to the right, it indicates that a few states have significantly lower literate populations.
"""

# Task (b): Construct scatter plot between male graduates and female graduates
plt.figure(figsize=(10, 6))
plt.scatter(indian_cities['male_graduates'], indian_cities['female_graduates'], color='coral', alpha=0.7)
plt.title('Scatter Plot between Male Graduates and Female Graduates in Indian Cities')
plt.xlabel('Male Graduates')
plt.ylabel('Female Graduates')
plt.grid(True)
plt.show()

"""
6)	 For the data set “Indian_cities”
a)	Construct Boxplot on total effective literacy rate and draw inferences
b)	Find out the number of null values in each column of the dataset and delete them.
"""

import seaborn as sns

# Task (a): Construct Boxplot on total effective literacy rate
plt.figure(figsize=(8, 6))
sns.boxplot(x=indian_cities['effective_literacy_rate_total'], color='lightgreen')
plt.title('Boxplot of Total Effective Literacy Rate in Indian Cities')
plt.xlabel('Effective Literacy Rate (Total)')
plt.show()

"""
Inference:
The interquartile range of total effect literacy rate of the cities lies between approximately 80 and 90. 
The cities with minimum value of the box plot is approximately 70 
There are few cities as outliers even below the min value of box plot.
"""

# Task (b): Find and remove null values
# Display the number of null values in each column
print("Number of null values in each column:")
print(indian_cities.isnull().sum())

# Remove rows with null values
indian_cities_cleaned = indian_cities.dropna()

# Display the number of null values after removal
print("\nNumber of null values in each column after removal:")
print(indian_cities_cleaned.isnull().sum())

