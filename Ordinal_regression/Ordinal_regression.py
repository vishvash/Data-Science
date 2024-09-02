# Ordinal Regression

# pip install mord

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Loading dataset
wvs = pd.read_csv("wvs.csv")
wvs.head()

# EDA
wvs.describe()
wvs.columns

#converting into binary
lb = LabelEncoder()
wvs["poverty"] = lb.fit_transform(wvs["poverty"])
wvs["religion"] = lb.fit_transform(wvs["religion"])
wvs["degree"] = lb.fit_transform(wvs["degree"])
wvs["country"] = lb.fit_transform(wvs["country"])
wvs["gender"] = lb.fit_transform(wvs["gender"])

# pip install mord
from mord import LogisticAT
model = LogisticAT(alpha = 0).fit(wvs.iloc[:, 1:], wvs.iloc[:, 0])  
# alpha parameter set to zero to perform no regularisation.fit(x_train,y_train)
model.coef_
model.classes_

predict = model.predict(wvs.iloc[:, 1:]) # Train predictions 

# Accuracy 
accuracy_score(wvs.iloc[:,0], predict)


