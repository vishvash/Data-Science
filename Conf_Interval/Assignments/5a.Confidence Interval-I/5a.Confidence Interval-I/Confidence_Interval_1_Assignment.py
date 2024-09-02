# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 08:21:07 2024

@author: Lenovo
"""

# Q1) Calculate probability from the given dataset for the below cases.

import pandas as pd
import statistics
import scipy.stats as stats

statistics.stdev

# Load the dataset
cars_data = pd.read_excel(r"C:\Users\Lenovo\Downloads\Study material\Data Science\Conf_Interval\Assignments\5a.Confidence Interval-I\5a.Confidence Interval-I\Cars.xlsx")

# Extract MPG data
mpg = cars_data['MPG']

statistics.stdev(mpg)
statistics.mean(mpg)

# a. P(MPG > 38)
prob_mpg_gt_38 = 1 - stats.norm.cdf(38, 34.42, 9.13)
print("Probability of MPG > 38:", prob_mpg_gt_38)

prob_mpg_lt_40 = stats.norm.cdf(40, 34.42, 9.13)
print("Probability of MPG < 40:", prob_mpg_lt_40)

# c. P(20 < MPG < 50)
prob_mpg_between_20_and_50 = stats.norm.cdf(50, 34.42, 9.13) - stats.norm.cdf(20, 34.42, 9.13)
print("Probability of 20 < MPG < 50:", prob_mpg_between_20_and_50)

#Q2) Check whether the data follows the normal distribution.
#a) Check whether the MPG of Cars follows the Normal Distribution Dataset: Cars.csv


from scipy.stats import shapiro

# Shapiro-Wilk test for normality
stat, p = shapiro(mpg)
print("Shapiro-Wilk Test for MPG of Cars:")
print("Test Statistic:", stat)
print("p-value:", p)
if p > 0.05:
    print("MPG of Cars follows a normal distribution.")
else:
    print("MPG of Cars does not follow a normal distribution.")
    
import seaborn as sns
import pylab
sns.kdeplot(mpg) 
stats.probplot(mpg, dist = "norm", plot = pylab)


# Load dataset
wc_at_data = pd.read_csv('wc-at.csv')

# Extract AT and Waist data
at = wc_at_data['AT']
waist = wc_at_data['Waist']

# Shapiro-Wilk test for normality for AT
stat_at, p_at = shapiro(at)
print("\nShapiro-Wilk Test for Adipose Tissue (AT):")
print("Test Statistic:", stat_at)
print("p-value:", p_at)
if p_at > 0.05:
    print("AT follows a normal distribution.")
else:
    print("AT does not follow a normal distribution.")

# Shapiro-Wilk test for normality for Waist
stat_waist, p_waist = shapiro(waist)
print("\nShapiro-Wilk Test for Waist Circumference:")
print("Test Statistic:", stat_waist)
print("p-value:", p_waist)
if p_waist > 0.05:
    print("Waist Circumference follows a normal distribution.")
else:
    print("Waist Circumference does not follow a normal distribution.")

sns.kdeplot(at) 
sns.kdeplot(waist) 
stats.probplot(at, dist = "norm", plot = pylab)
stats.probplot(waist, dist = "norm", plot = pylab)


import scipy.stats as stats

# For Q3
# Z-scores
z_90 = stats.norm.ppf(0.95)
z_94 = stats.norm.ppf(0.97)
z_60 = stats.norm.ppf(0.8)

# For Q4
# Sample size
n = 25

# t-scores
t_95 = stats.t.ppf(0.975, df=n-1)
t_96 = stats.t.ppf(0.98, df=n-1)
t_99 = stats.t.ppf(0.995, df=n-1)

print("Q3) Z-scores:")
print("For 90% confidence interval:", z_90)
print("For 94% confidence interval:", z_94)
print("For 60% confidence interval:", z_60)

print("\nQ4) t-scores:")
print("For 95% confidence interval:", t_95)
print("For 96% confidence interval:", t_96)
print("For 99% confidence interval:", t_99)


from scipy.stats import t

mean_population = 270
sample_mean = 260
sample_std = 90
n = 18

# Calculate t-score
t_score = (sample_mean - mean_population) / (sample_std / (n ** 0.5))

# Calculate probability
p_value = t.cdf(t_score, df=n-1)

print("Probability that 18 randomly selected bulbs would have an average life of no more than 260 days:", p_value)


from scipy.stats import norm

mu = 45  # Mean
sigma = 8  # Standard Deviation

# Probability that service time > 60 minutes
p_service_time_gt_50 = 1 - norm.cdf(50, mu, sigma)

print("Probability that the service manager cannot meet his commitment:", p_service_time_gt_50)


from scipy.stats import norm

mu = 38
sigma = 6

# Probability of employees being older than 44
prob_gt_44 = 1 - norm.cdf(44, loc=mu, scale=sigma)

# Probability of employees being between 38 and 44
prob_between_38_and_44 = norm.cdf(44, loc=mu, scale=sigma) - norm.cdf(38, loc=mu, scale=sigma)

# Compare the probabilities
if prob_gt_44 > prob_between_38_and_44:
    print("True. More employees at the processing center are older than 44 than between 38 and 44.")
else:
    print("False. More employees at the processing center are not older than 44 than between 38 and 44.")


# Calculate z-score for X = 30
z_score_30 = (30 - mu) / sigma

# Find the percentage of employees under the age of 30
percent_under_30 = norm.cdf(30, loc=mu, scale=sigma) * 100

print("Percentage of employees under the age of 30:", percent_under_30, "%")

# Calculate the number of employees
total_employees = 400
employees_under_30 = total_employees * (percent_under_30 / 100)

print("Expected number of employees under the age of 30:", round(employees_under_30))


from scipy.stats import norm

mean = 100
std_dev = 20

# Find the z-score for the given probabilities
z_score_lower = norm.ppf(0.005, 0, 1)
z_score_upper = norm.ppf(0.995, 0, 1)

# Convert z-scores to values
a = mean + z_score_lower * std_dev
b = mean + z_score_upper * std_dev

print("Values a and b symmetric about the mean:", a, b)


# Given data
mean_profit1 = 5 
std_dev_profit1 = 3  
mean_profit2 = 7 
std_dev_profit2 = 4  
conversion_rate = 45 

total_mean_profit = mean_profit1 + mean_profit2

# Calculate total standard deviation of profit (sqrt(variance) = sqrt(std_dev^2))
total_std_dev_profit = ((std_dev_profit1 ** 2) + (std_dev_profit2 ** 2)) ** 0.5

# Calculate the range containing 95% probability (95% confidence interval)
# For a normal distribution, 95% probability lies within 1.96 standard deviations from the mean
lower_bound = total_mean_profit - 1.96 * total_std_dev_profit
upper_bound = total_mean_profit + 1.96 * total_std_dev_profit
lower_bound_cr_rs = round((lower_bound * 4.5),2)
upper_bound_cr_rs = round((upper_bound * 4.5),2)

print("Rupee range (centered on the mean) containing 95% probability for the annual profit of the company:")
print("Lower bound: "+ str(lower_bound_cr_rs) +" Crores")
print("Upper bound: "+ str(upper_bound_cr_rs) +" Crores")


from scipy.stats import norm

# Calculate the 5th percentile of profit (in Rupees)
percentile_5_rs = norm.ppf(0.05, loc=total_mean_profit, scale=total_std_dev_profit)

print("5th percentile of profit (in Crores) for the company:", round((percentile_5_rs * 4.5),2))


# Calculate the probability of making a loss for each division
prob_loss_profit1 = norm.cdf(0, loc=mean_profit1, scale=std_dev_profit1)
prob_loss_profit2 = norm.cdf(0, loc=mean_profit2, scale=std_dev_profit2)

# Compare the probabilities
if prob_loss_profit1 > prob_loss_profit2:
    print("Profit1 has a larger probability of making a loss each year.")
elif prob_loss_profit1 < prob_loss_profit2:
    print("Profit2 has a larger probability of making a loss each year.")
else:
    print("Both divisions have the same probability of making a loss each year.")
