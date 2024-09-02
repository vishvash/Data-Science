# Problem Statement

'''
Government initiative focused on improving employability among communities 
living below the poverty line through self-employment schemes and skill training 
programs. This approach, coupled with additional benefit schemes to motivate
participation, reflects a comprehensive strategy to address unemployment and
underemployment in disadvantaged communities.

Such initiatives, if well-implemented, can have a significant impact not just 
on individual livelihoods but also on the broader economic and social development 
of the community.

Objective: Minimize the unemployment ratio.
Constraints: Improve the overall impact on the community.

'''
# pip install lifelines

import pandas as pd
# Loading the the survival un-employment data
survival_unemp = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Survival_Analysis/Survival_Analysis/survival_unemployment.csv")
survival_unemp.head()
survival_unemp.describe()

survival_unemp["spell"].describe()

# Spell is referring to time 
T = survival_unemp.spell

# Importing the KaplanMeierFitter model to fit the survival analysis
from lifelines import KaplanMeierFitter

# Initiating the KaplanMeierFitter model
kmf = KaplanMeierFitter()

# Fitting KaplanMeierFitter model on Time and Events for death 
kmf.fit(T, event_observed = survival_unemp.event)

# Time-line estimations plot 
kmf.plot()

# Over Multiple groups
# For each group, here group is ui
survival_unemp.ui.value_counts()

# Applying KaplanMeierFitter model on Time and Events for the group "1"
kmf.fit(T[survival_unemp.ui == 1], survival_unemp.event[survival_unemp.ui == 1], label = '1')
ax = kmf.plot()

# Applying KaplanMeierFitter model on Time and Events for the group "0"
kmf.fit(T[survival_unemp.ui == 0], survival_unemp.event[survival_unemp.ui == 0], label = '0')
kmf.plot(ax=ax)
