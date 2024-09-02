import pandas as pd 
from statsmodels.formula.api import poisson
from statsmodels.formula.api import negativebinomial


claims = pd.read_csv('C:/Users/Lenovo/Downloads/Study material/Data Science/Advanced Regression for Count Data/Poisson  NBReg_claims/claims.csv')
claims.head(5)
claims = claims.iloc[:, 1:] # Excluding id column

claims.describe()

claims['numclaims'].unique()
claims.numclaims.value_counts()


# poisson Regression
m1 = poisson('numclaims ~ numveh + age', data = claims).fit()
print(m1.summary())

preds_pos = m1.predict()

model_fit1 = claims
model_fit1['pred_claims'] = preds_pos
model_fit1


# Negative Binomial regression
m_nb = negativebinomial('numclaims ~ numveh + age', data = claims).fit()
print(m_nb.summary())

preds_nb = m_nb.predict()
model_fit2 = claims
model_fit2['nb_predicted'] = preds_nb
model_fit2
