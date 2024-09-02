# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 14:03:39 2024

@author: Lenovo
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import statsmodels.stats.descriptivestats as sd
from statsmodels.stats import weightstats as stests 

'''1.	A F&B manager wants to determine whether there is any significant difference in the diameter of the cutlet between two units. A randomly selected sample of cutlets was collected from both units and measured. Analyze the data and draw inferences at a 5% significance level. Please state the assumptions and tests that you carried out to check the validity of the assumptions'''

############ 2-sample t-Test ##################
# Business Problem:
# Determine whether there is any significant difference in the diameter of the cutlet between two units

# Load the data
cutlet = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Hypothesis Testing/Assignments/Datasets/Cutlets.csv")
cutlet

cutlet.columns = "Unit_A", "Unit_B"

# Normality Test - # Shapiro Test
# H0 = Data are Normal
# Ha = Data are not Normal
stats.shapiro(cutlet.Unit_A) # Shapiro Test
print(stats.shapiro(cutlet.Unit_B))
# Data are Normal
# help(stats.shapiro)

# Variance test
# H0 = Variances are Equal
# Ha = Variance are not Equal
scipy.stats.levene(cutlet.Unit_A, cutlet.Unit_B)

# p-value = 0.417 > 0.05 so p high null fly => Equal variances

# 2 Sample T test
# Ho: Average diameter of both cutlets are equal 
# Ha: Average diameter of both cutlets are unequal 

scipy.stats.ttest_ind(cutlet.Unit_A, cutlet.Unit_B)

# p-value = 0.47 > 0.05 
# P high Ho fly => Fail to reject Ho

# Conclusion: # Average diameter of both cutlets are equal 

'''2.	A hospital wants to determine whether there is any difference in the average Turn Around Time (TAT) of reports of the laboratories on their preferred list. They collected a random sample and recorded TAT for reports from 4 laboratories. TAT is defined as a sample collected to report dispatch. Analyze the data and determine whether there is any difference in average TAT among the different laboratories at a 5% significance level. 
Data File: LabTAT.csv
'''

############# One-Way ANOVA #############
# Business Problem: 
# Hospital to determine whether there is any difference in average TAT among the different laboratories at a 5% significance level

lab_tat = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Hypothesis Testing/Assignments/Datasets/lab_tat_updated.csv")
lab_tat

# Normality Test - # Shapiro Test
# H0 = Data are Normal
# Ha = Data are not Normal
stats.shapiro(lab_tat.Laboratory_1)
stats.shapiro(lab_tat.Laboratory_2)
stats.shapiro(lab_tat.Laboratory_3)
stats.shapiro(lab_tat.Laboratory_4)


# Variance test
# Ho: All the 4 Laboratories have equal variance TAT 
# Ha: All the 4 Laboratories have unequal variance TAT 
scipy.stats.levene(lab_tat.Laboratory_1, lab_tat.Laboratory_2, lab_tat.Laboratory_3, lab_tat.Laboratory_4)
# Variances are statisticaly equal

# One - Way Anova
# Ho: All the 4 Laboratories have equal mean TAT 
# Ha: All the 4 Laboratories have unequal mean TAT 

F, p = stats.f_oneway(lab_tat.Laboratory_1, lab_tat.Laboratory_2, lab_tat.Laboratory_3, lab_tat.Laboratory_4)

p 
# P low Null go p =2.143740909435053e-58 < 0.05
# All the 4 Laboratories have unequal mean TAT 

#Conclusion: There is any difference in average TAT among the different laboratories at a significance level


'''3.	Sales of products in four different regions are tabulated for males and females. Find if male-female buyer rations are similar across regions.

	East	West	North	South
Males	50	142	131	70
Females	550	351	480	350

•	Ho   All proportions are equal
•	Ha   Not all Proportions are equal
Hint: 
Check p-value
If p-Value < alpha, we reject Null Hypothesis
Data file: Buyer Ratio.csv
'''

############### Chi-Square Test ################

# Business Problem: 
# Has to determine if male-female buyer ratios are similar across regions

data = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Hypothesis Testing/Assignments/Datasets/BuyerRatio.csv")
data = data.iloc[:,1:]

# Ho: All regions have equal proportions of male-female buyer %
# Ha: Not all regions have equal proportions of male-female buyer %
Chisquares_results = scipy.stats.chi2_contingency(data)

print(Chisquares_results)

Chi_square = [['Test Statistic', 'p-value'], [Chisquares_results[0], Chisquares_results[1]]]
Chi_square
# p-value = 0.66 > 0.05 => P hight Null fly

# All regions have equal proportions of male-female buyer %
# Conclusion: All regions have equal proportions of male-female buyer %


'''4.	Telecall uses 4 centers around the globe to process customer order forms. They audit a certain % of the customer order forms. Any error in the order form renders it defective and must be reworked before processing. The manager wants to check whether the defective % varies by center. Please analyze the data at a 5% significance level and help the manager draw appropriate inferences.
            File: Customer OrderForm.csv
'''

############### Chi-Square Test ################
# Business Problem: 
# Check whether the defective % varies by center. 

original_df = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Hypothesis Testing/Assignments/Datasets/CustomerOrderform.csv")
original_df

original_df.replace({'Error Free': 0, 'Defective': 1}, inplace=True)

# Apply value_counts to each column
result_df = original_df.apply(lambda x: x.value_counts()).fillna(0)

# Print the result
print(result_df)

# Ho: All centres have equal proportions of defectives %
# Ha: Not all centres have equal proportions of defectives %
Chisquares_results = scipy.stats.chi2_contingency(result_df)

print(Chisquares_results)

Chi_square = [['Test Statistic', 'p-value'], [Chisquares_results[0], Chisquares_results[1]]]
Chi_square
# p-value = 0.28 > 0.05 => P high Null fly

# All centres have equal proportions 
# Conclusion: All Proportions are equal 


'''5.	Fantaloons Sales managers commented that % of males versus females walking into the store differs based on the day of the week. Analyze the data and determine whether there is evidence at a 5 % significance level to support this hypothesis. 
File: Fantaloons.csv
'''

######### 2-Proportion Test #########

# Business Problem: 
# Sales manager has to determine if % of males versus females walking into the store differs based on the day of the week
two_prop_test = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Hypothesis Testing/Assignments/Datasets/Fantaloons.csv")

from statsmodels.stats.proportion import proportions_ztest

tab1 = two_prop_test.Weekdays.value_counts()
print(tab1)

tab2 = two_prop_test.Weekend.value_counts()
print(tab2)

count = np.array([113, 167])
nobs = np.array([400, 400])


# Case1: Two Sided test
# Ho: Males-Female ratio is same on weekdays and on weekend
# Ha: Males-Female ratio is not same on weekdays and on weekend

stats, pval = proportions_ztest(count, nobs, alternative = 'two-sided') 
print("%.2f" % pval)
# P-value = 0.000 < 0.05 => P low Null go
# Ha: Males-Female ratio is not same on weekdays and on weekend

# Case2: One-sided (Greater) test
# Ho: Proportions of Males in weekdays <= Proportions of Males in weekend
# Ha: Proportions of Males in weekdays > Proportions of Males in weekend
stats, pval = proportions_ztest(count, nobs, alternative = 'larger')
print("%.2f" % pval) 
# P-value = 1.0 > 0.05 => P high Null fly

# Ho: Proportions of Males in weekdays <= Proportions of Males in weekend
# Conclusion: % of males versus females walking into the store differs based on the day of the week
