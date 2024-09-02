'''
# CRISP-ML(Q):

Business Problem: There are a lot of assumptions in the diagnosis pertaining to cancer.
In a few cases radiologists, pathologists and oncologists go wrong in diagnosing whether 
tumor is benign (non-cancerous) or malignant (cancerous). 
Hence team of physicians want us to build an AI application which will predict with 
confidence the presence of cancer in a patient. This will serve as a compliment to the physicians.

Business Objective: Maximize Cancer Detection

Business Constraints: Minimize Treatment Cost & Maximize Patient Convenience

Success Criteria:

Business success criteria: Increase the correct diagnosis of cancer in at least 96% of patients
Machine Learning success criteria: Achieve an accuracy of atleast 98%
Economic success criteria: Reducing medical expenses will improve trust of patients and 
thereby hospital will see an increase in revenue by atleast 12%

Data Collection:

Data is collected from the hospital for 569 patients. 30 features and 1 label 
comprise the feature set. Ten real-valued features are computed for each cell nucleus:

a) radius (mean of distances from center to points on the perimeter)
b) texture (standard deviation of gray-scale values)
c) perimeter
d) area
e) smoothness (local variation in radius lengths)
f) compactness (perimeter^2 / area - 1.0)
g) concavity (severity of concave portions of the contour)
h) concave points (number of concave portions of the contour)
i) symmetry
j) fractal dimension ("coastline approximation" - 1)
'''


# Import the required libraries
from sklearn import datasets, linear_model, neighbors, ensemble #, svm, naive_bayes
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import VotingClassifier
# from sklearn.metrics import accuracy_score

import numpy as np
import pickle

# Load the dataset
breast_cancer = datasets.load_breast_cancer()
breast_cancer

# Input and Output
X, y = breast_cancer.data, breast_cancer.target


# Split the train and test samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y, random_state = 123)


# Base Model 1
### k-Nearest Neighbors (k-NN) with GridSearchCV
knn = neighbors.KNeighborsClassifier()

params_knn = {'n_neighbors': np.arange(1, 25)}

knn_gs = GridSearchCV(knn, params_knn, cv = 5)

knn_gs.fit(X_train, y_train)
knn_gs.best_params_

knn_best = knn_gs.best_estimator_



# Base Model 2
### Random Forest Classifier with GridSearchCV
rf = ensemble.RandomForestClassifier(random_state = 0)

params_rf = {'n_estimators': [50, 100, 200]}

rf_gs = GridSearchCV(rf, params_rf, cv = 5)

rf_gs.fit(X_train, y_train)
rf_gs.best_params_

rf_best = rf_gs.best_estimator_



# Base Model 3
### Logistic Regression with GridSearchCV
log_reg = linear_model.LogisticRegression()

C = np.logspace(1, 4, 10)
params_lr = dict(C = C)

lr_gs = GridSearchCV(log_reg, params_lr, cv = 5)

lr_gs.fit(X_train, y_train)
lr_gs.best_estimator_

lr_best = lr_gs.best_estimator_



# Combine all three Based models
estimators = [('knn', knn_best), ('rf', rf_best), ('log_reg', lr_best)]



# Hard/Majority Voting
# # VotingClassifier with voting = "hard" parameter
ensemble_H = VotingClassifier(estimators, voting = "hard")

# Fit classifier with the training data
hard_voting = ensemble_H.fit(X_train, y_train)

# Save the voting classifier
pickle.dump(hard_voting, open('hard_voting.pkl', 'wb'))

# Loading a saved model
model = pickle.load(open('hard_voting.pkl', 'rb'))

print("knn_gs.score: ", knn_best.score(X_test, y_test))
# Output: knn_gs.score:
    
print("rf_gs.score: ", rf_best.score(X_test, y_test))
# Output: rf_gs.score:

print("log_reg.score: ", lr_best.score(X_test, y_test))
# Output: log_reg.score:

# Hard Voting Ensembler
print("Hard Voting Ensemble Score: ", ensemble_H.score(X_test, y_test))
# Output: ensemble.score:



#############################################################
# Soft Voting
# VotingClassifier with voting = "soft" parameter
ensemble_S = VotingClassifier(estimators, voting = "soft")

soft_voting = ensemble_S.fit(X_train, y_train)

# Save model
pickle.dump(soft_voting, open('soft_voting.pkl', 'wb'))


# Load the saved model
model = pickle.load(open('soft_voting.pkl', 'rb'))

print("knn_gs.score: ", knn_gs.score(X_test, y_test))
# Output: knn_gs.score:

print("rf_gs.score: ", rf_gs.score(X_test, y_test))
# Output: rf_gs.score:

print("log_reg.score: ", lr_gs.score(X_test, y_test))
# Output: log_reg.score:

print("Soft Voting Ensemble Score: ", ensemble_S.score(X_test, y_test))
# Output: ensemble.score:

