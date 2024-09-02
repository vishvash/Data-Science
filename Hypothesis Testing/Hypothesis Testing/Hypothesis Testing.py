'''
# 1 Sample Z-Test
# 1 Sample T-Test
# 1 Sample Sign Test

# Mann-Whitney test
# Paired T-Test
# 2 sample T-Test

# Moods-Median Test
# One-Way Anova

# 1-Proportion Test
# 2-Proportion Test
# Chi-Square Test
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import statsmodels.stats.descriptivestats as sd
from statsmodels.stats import weightstats as stests


############# 1-Sample Z-Test #############
# Business Problem: 
#    Verify if the length of the fabric is being cut at appropriate sizes (lengths)

#  importing the data
fabric = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Hypothesis Testing/Hypothesis Testing/Fabric_data.csv")

# Calculating the normality test
# Hypothesis
# Ho = Data are Normal
# Ha = Data are not Normal

print(stats.shapiro(fabric.Fabric_length))
# p-value = 0.146 > 0.05 so p high null fly => Data are Normal

# Calculating the mean
np.mean(fabric.Fabric_length)

# Population standard deviation (Sigma) is known
# z-test
# Ho: Current Mean is Equal to Standard Mean (150) => No action
# Ha: Current Mean is Not Equal to Standard Mean (150) => Take action

# parameters in z-test, value is mean of data
ztest, pval = stests.ztest(fabric.Fabric_length, x2 = None, value = 150)

print(float(pval))

# p-value = 7.156e-06 < 0.05 so p low null go

# z-test
# parameters in z-test, value is mean of data
# z-test
# Ho: Current Mean <= Standard Mean (150) => No action
# Ha: Current Mean > Standard Mean (150) => Take action

ztest, pval = stests.ztest(fabric.Fabric_length, x2 = None, value = 150, alternative = 'larger')

print(float(pval))

# p-value = 3.578-06 < 0.05 => p low null go

# Conclusion: Stop the production and verify the problem with the machine


############# 1-Sample t-Test #############
# Business Problem: 
#     Verify if the monthly energy cost differs from $200.

# loading the csv file
data = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Hypothesis Testing/Hypothesis Testing/FamilyEnergyCost.csv")
  
data.describe()

# Normality test
# Hypothesis
# Ho = Data are Normal
# Ha = Data are not Normal

print(stats.shapiro(data['Energy Cost']))
# p-value = 0.764 > 0.05 so p high null fly => Data are Normal

# Population standard deviation (Sigma) is not known

# Perform one sample t-test
# Ho: The monthly average energy cost for families is equal to $200
# Ha: The monthly average energy cost for families is not equal to $200

t_statistic, p_value = stats.ttest_1samp(a = data['Energy Cost'], popmean = 200)
print("%.2f" % p_value)

# Ho: Current Mean <= Standard Mean (200)
# Ha: Current Mean > Standard Mean (200)

t_statistic, p_value = stats.ttest_1samp(a = data['Energy Cost'], popmean = 200, alternative = 'greater')
print("%.2f" % p_value)

# p-value = 0.00 < 0.05 => p low null go

# Conclusion: The average monthly energy cost for families is greater than $200. So reduce electricity consumption


############ Non-Parameteric Test ############

############ 1 Sample Sign Test ################

# Stainless-Steel Chromium content
steel = pd.read_excel(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Hypothesis Testing/Hypothesis Testing/StainlessSteelComposition.xlsx")
steel

# Note: Most stainless steel contains about 18 percent chromium; 
# it is what hardens and toughens steel and increases its resistance 
# to corrosion, especially at high temperatures. 

# Business Problem: 
#    Determine whether the median chromium content differs from 18%.

# Normality Test
# Ho = Data are Normal
# Ha = Data are not Normal
stats.shapiro(steel.Chromium) # Shapiro Test

# p-value = 0.00016 < 0.05 so p low null go => Data are not Normal

### 1 Sample Sign Test ###

# Ho: The median chromium content is equal to 18%.
# Ha: The median chromium content is not equal to 18%.

sd.sign_test(steel.Chromium, mu0 = 18)
# sd.sign_test(marks.Scores, mu0 = marks.Scores.median())

# Conclusion: Enough evidence to conclude that the median chromium content is equal to 18%.



######### Mann-Whitney Test ############
# Vehicles with and without additive

# Business Problem: 
#    Fuel additive is enhancing the performance (mileage) of a vehicle

fuel = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Hypothesis Testing/Hypothesis Testing/mann_whitney_additive.csv")
fuel

fuel.columns = "Without_additive", "With_additive"

# Normality test 
# Ho = Data are Normal
# Ha = Data are not Normal

print(stats.shapiro(fuel.Without_additive))  # p high null fly
print(stats.shapiro(fuel.With_additive))     # p low null go

# Data are not normal

# Non-Parameteric Test case
# Mann-Whitney test

# Ho: Mileage with and without Fuel additive is the same
# Ha: Mileage with and without Fuel additive are different

scipy.stats.mannwhitneyu(fuel.Without_additive, fuel.With_additive)

# p-value = 0.44 > 0.05 so p high null fly
# Ho: fuel additive does not impact the performance


############### Paired T-Test ##############
# A test to determine whether there is a significant difference between 2 variables.

# Data:
#  Data shows the effect of two soporific drugs
# (increase in hours of sleep compared to control) on 10 patients.
# External Conditions are conducted in a controlled environment to ensure external conditions are same

# Business Problem: 
#   Determine which of the two soporific drugs increases the sleep duration

sleep = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Hypothesis Testing/Hypothesis Testing/sleep.csv")
sleep.describe()

# Normality Test - # Shapiro Test
# H0 = Data are Normal
# Ha = Data are not Normal

stats.shapiro(sleep.extra[0:10]) 
stats.shapiro(sleep.extra[10:20])
# Data are Normal

# Assuming the external Conditions are same for both the samples
# Paired T-Test
# Ho: Increase in the sleep with Drug 1 = Increase in the sleep with Drug 2
# Ha: Increase in the sleep with Drug 1 != Increase in the sleep with Drug 2

ttest, pval = stats.ttest_rel(sleep.extra[0:10], sleep.extra[10:20])
print(pval)
# p-value = 0.002 < 0.05 => p low null go
# Ha: Increase in the sleep with Drug 1 != Increase in the sleep with Drug 2


# Ho: Increase in the sleep with Drug 1 <= Increase in the sleep with Drug 2
# Ha: Increase in the sleep with Drug 1 > Increase in the sleep with Drug 2
ttest, pval = stats.ttest_rel(sleep.extra[0:10], sleep.extra[10:20], alternative = 'greater')
print(pval)


# p-value = 0.99 > 0.05 => p high null fly
# Ho: Increase in the sleep with Drug 1 <= Increase in the sleep with Drug 2



############ 2-sample t-Test (Marketing Strategy) ##################
# Business Problem:
# Determine whether the Full Interest Rate Waiver is better than Standard Promotion

# Load the data
prom = pd.read_excel(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Hypothesis Testing/Hypothesis Testing (1)/Promotion.xlsx")
prom

prom.columns = "InterestRateWaiver", "StandardPromotion"

# Normality Test - # Shapiro Test
# H0 = Data are Normal
# Ha = Data are not Normal
stats.shapiro(prom.InterestRateWaiver) # Shapiro Test
print(stats.shapiro(prom.StandardPromotion))
# Data are Normal
help(stats.shapiro)

# Variance test
# H0 = Variances are Equal
# Ha = Variance are not Equal
scipy.stats.levene(prom.InterestRateWaiver, prom.StandardPromotion)

# p-value = 0.287 > 0.05 so p high null fly => Equal variances

# 2 Sample T test
# Ho: Average purchases because of both promotions are equal 
# Ha: Average purchases because of both promotions are unequal 

scipy.stats.ttest_ind(prom.InterestRateWaiver, prom.StandardPromotion)

# p-value = 0.024 < 0.05 
# P low Ho Go => Fail to accept Ho

scipy.stats.ttest_ind(prom.InterestRateWaiver, prom.StandardPromotion, alternative = 'greater')
# Ho: Average purchases because of InterestRateWaiver <= Average purchases because of StandardPromotion
# Ha: Average purchases because of InterestRateWaiver > Average purchases because of StandardPromotion
# p-value = 0.012 < 0.05 so p low null go

# Conclusion: # Interest Rate Waiver is a better promotion than Standard Promotion


###### Moods-Median Test ######
# Business Problem: 
# Determine if the weight of the fish differs with the change in the temperatures.

# Import dataset
fish = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Hypothesis Testing/Hypothesis Testing/Fishweights.csv")
fish.describe()

# 4 groups 
g1 = fish[fish.group == 1]
g2 = fish[fish.group == 2]
g3 = fish[fish.group == 3]
g4 = fish[fish.group == 4]

# Normality Test - # Shapiro Test
# Ho = Data are Normal
# Ha = Data are not Normal
stats.shapiro(g1)
stats.shapiro(g2)
stats.shapiro(g3)
stats.shapiro(g4)

# Mood's median test compares medians among two or more groups.
# Ho: The population medians are equal across fish groups.
# Ha: The population medians are not equal across fish groups.

from scipy.stats import median_test
stat, p, med, tbl = median_test(g1.Weight, g2.Weight, g3.Weight, g4.Weight)

p 
# 0.696 > 0.05 => P High Null Fly

# Fail to reject the null hypothesis
# The differences between the median weights are not statistically significant.
# Further tests must be done to determine which fish weight is more than the rest, which is out of scope for our discussion. 

############# One-Way ANOVA #############
# Business Problem: 
# CMO to determine the renewal of contracts of the suppliers based on their performances

con_renewal = pd.read_excel(r"C:\Users\Lenovo\Downloads\Study material\Data Science\Hypothesis Testing\Hypothesis Testing\ContractRenewal_Data(unstacked).xlsx")
con_renewal
con_renewal.columns = "SupplierA", "SupplierB", "SupplierC"

# Normality Test - # Shapiro Test
# H0 = Data are Normal
# Ha = Data are not Normal
stats.shapiro(con_renewal.SupplierA)
stats.shapiro(con_renewal.SupplierB)
stats.shapiro(con_renewal.SupplierC)


# Variance test
# Ho: All the 3 suppliers have equal variance transaction time
# Ha: All the 3 suppliers have unequal variance transaction time
scipy.stats.levene(con_renewal.SupplierA, con_renewal.SupplierB, con_renewal.SupplierC)
# Variances are statisticaly equal

# One - Way Anova
# Ho: All the 3 suppliers have equal mean transaction time
# Ha: All the 3 suppliers have unequal mean transaction time

F, p = stats.f_oneway(con_renewal.SupplierA, con_renewal.SupplierB, con_renewal.SupplierC)

p 
# P High Null Fly
# All the 3 suppliers have equal mean transaction time

######### 1-Proportion Test #########
# Business Problem: 
# The proportion of smokers is varying with the historical figure of 25 percent of students who smoke regularly.

smokers = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Hypothesis Testing/Hypothesis Testing/Smokers.csv")
smokers.head()

# Breakup of smokers and non-smokers
obs = smokers['Smokes'].value_counts()
obs

# Number of smokers
x = obs[1]
x

# Total observations
n = len(smokers)
n

# Ho: Proportion of students who smoke regularly is less than or equal to the historcal figure 25%
# Ha: Proportion of students who smoke regularly is greater than the historcal figure 25%
from statsmodels.stats.proportion import proportions_ztest
stats, pval = proportions_ztest(count = x, nobs = n, value = 0.25, alternative = 'two-sided')  

print("%.2f" % pval)
# 0.33 > 0.05 => P high Null fly => Ho

# Ho: Proportion of students who smoke regularly has not increased beyond 25%.

######### 2-Proportion Test #########

# Business Problem: 
# Sales manager has to determine if the sales incentive program should be launced or not

import numpy as np

two_prop_test = pd.read_excel(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Hypothesis Testing/Hypothesis Testing/JohnyTalkers.xlsx")

from statsmodels.stats.proportion import proportions_ztest

tab1 = two_prop_test.Person.value_counts()
print(tab1)

tab2 = two_prop_test.Drinks.value_counts()
print(tab2)


# crosstable table
pd.crosstab(two_prop_test.Person, two_prop_test.Drinks)

count = np.array([58, 152])
nobs = np.array([480, 740])


# Case1: Two Sided test
# Ho: Proportions of Adults = Proportions of Children
# Ha: Proportions of Adults != Proportions of Children

stats, pval = proportions_ztest(count, nobs, alternative = 'two-sided') 
print("%.2f" % pval)
# P-value = 0.000 < 0.05 => P low Null go
# Ha: Proportions of Adults != Proportions of Children

# Case2: One-sided (Greater) test
# Ho: Proportions of Adults <= Proportions of Children
# Ha: Proportions of Adults > Proportions of Children
stats, pval = proportions_ztest(count, nobs, alternative = 'larger')
print("%.2f" % pval) 
# P-value = 1.0 > 0.05 => P high Null fly

# Ho: Proportions of Adults <= Proportions of Children
# Conclusion: Do not launch incentive program


############### Chi-Square Test ################
# Business Problem: 
# Check whether the questionnaire responses input entry defectives % varies by region. 

Bahaman = pd.read_excel(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Hypothesis Testing/Hypothesis Testing/Bahaman.xlsx")
Bahaman

count = pd.crosstab(Bahaman["Defective"], Bahaman["Country"])
count

# Ho: All countries have equal proportions of defectives %
# Ha: Not all countries have equal proportions of defectives %
Chisquares_results = scipy.stats.chi2_contingency(count)

print(Chisquares_results)

Chi_square = [['Test Statistic', 'p-value'], [Chisquares_results[0], Chisquares_results[1]]]
Chi_square
# p-value = 0.63 > 0.05 => P high Null fly

# All countries have equal proportions 
# Conclusion: All Proportions are equal 


### The End