# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 19:43:53 2024

@author: Lenovo
"""

import scipy.stats as stats

# Given data
sample_mean = 200  # in pounds
sample_std_dev = 30  # in pounds
sample_size = 2000

# Calculate confidence intervals
conf_levels = [0.94, 0.98, 0.96]

for conf_level in conf_levels:
    # Find the critical value for the t-distribution
    t_value = stats.t.ppf((1 + conf_level) / 2, sample_size - 1)
    # Calculate margin of error
    margin_of_error = t_value * (sample_std_dev / (sample_size ** 0.5))
    # Calculate confidence interval
    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error
    print(f"{conf_level * 100}% confidence interval: ({lower_bound}, {upper_bound})")

import math

# Given data
Z = stats.norm.ppf(0.975)  # Z-score for 95% confidence level
p = 0.5  # Assuming maximum variability
E = 0.04  # Margin of error

# Calculate minimum sample size
minimum_sample_size = math.floor((Z ** 2 * p * (1 - p)) / E ** 2)
print("Q9: Minimum number of employers:", minimum_sample_size)


import scipy.stats as stats
import math

# Given data
confidence_level = 0.98  # 98% confidence level
Z = stats.norm.ppf(0.99)  # Z-score for 98% confidence level
p = 0.5  # Assumed proportion for maximum variability
margin_of_error = 0.04  # Desired margin of error

# Calculate minimum sample size
minimum_sample_size = math.floor((Z ** 2 * p * (1 - p)) / margin_of_error ** 2)
print("Minimum sample size:", minimum_sample_size)


