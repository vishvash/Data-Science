# Problem Statement

'''
Problem Statement: ECG of different pericardialeffusion groups of people has been recorded. The survival time in hours after the operation is given and the event type is denoted by 1 (if dead) and 0 (if alive). Perform survival analysis on the dataset given below and provide your insights in the documentation.

Objective: Maximize the survival rate.
Constraints: Improve the overall treatmet for the patient.

'''
# pip install lifelines

import pandas as pd
# Loading the the survival un-employment data
survival_dead = pd.read_excel(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Survival_Analysis/Assignment/Survival Analytics/ECG_Surv.xlsx")
survival_dead.head()
survival_dead.describe()

survival_dead["survival_time_hr"].describe()

# survival_time_hr is referring to time 
T = survival_dead.survival_time_hr

# Importing the KaplanMeierFitter model to fit the survival analysis
from lifelines import KaplanMeierFitter

# Initiating the KaplanMeierFitter model
kmf = KaplanMeierFitter()

# Fitting KaplanMeierFitter model on Time and Events for death 
kmf.fit(T, event_observed = survival_dead.alive)

# Time-line estimations plot 
kmf.plot()


# Over Multiple groups
# For each group, here group is pericardialeffusion
survival_dead.pericardialeffusion.value_counts()

# Applying KaplanMeierFitter model on Time and Events for the group "0"
kmf.fit(T[survival_dead.pericardialeffusion == 0], survival_dead.alive[survival_dead.pericardialeffusion == 0], label = 'Absence of pericardial effusion')
ax = kmf.plot()

# Applying KaplanMeierFitter model on Time and Events for the group "1"
kmf.fit(T[survival_dead.pericardialeffusion == 1], survival_dead.alive[survival_dead.pericardialeffusion == 1], label = 'Presence of pericardial effusion')

kmf.plot(ax=ax)

'''
If there is Presence of pericardial effusion the probability of patient dies increases after the surgery.
Whereas if there is no presence of pericardical effusion the probability of patient dies is comparitively lesser.
'''