'''CRISP-ML(Q)

a. Business & Data Understanding
    As the economy becomes more competitive, companies are increasingly relying on data-driven approaches to optimize their operations and make strategic decisions. One such area is predicting salary levels based on various demographic and job-related factors. Understanding the determinants of salary levels can help companies attract and retain talent, optimize compensation structures, and ensure fair pay practices.

    i. Business Objective - Optimize Salary Prediction
    ii. Business Constraint - Minimize Prediction Errors

    Success Criteria:
    1. Business Success Criteria - Increase employee satisfaction by 15% by accurately predicting salary levels and ensuring fair pay practices.
    2. ML Success Criteria - Achieve a prediction accuracy of atleast by 80% and performance of predicting salary levels for new data.
    3. Economic Success Criteria - Reduce turnover costs by accurately predicting salary levels and offering competitive compensation packages. By optimizing salary prediction, companies can reduce turnover rates by atleast 20 %.

    Data Collection - Salary data from various companies and industries is collected, including information on age, education, occupation, work experience, and other relevant factors. The dataset contains 15 features and the target variable is salary level.

    Metadata Description:
    Feature Name       Description                                  Type          Relevance
    --------------     -----------------------------------------     ------------- --------------
    age               Age of the individual                        Quantitative  Relevant
    workclass         Type of work class                           Nominal       Relevant
    education         Level of education                           Nominal       Relevant
    educationno       Number of years of education                 Quantitative  Relevant
    maritalstatus     Marital status                               Nominal       Relevant
    occupation        Occupation                                   Nominal       Relevant
    relationship      Relationship status                          Nominal       Relevant
    race              Race of the individual                       Nominal       Relevant
    sex               Gender                                       Nominal       Relevant
    capitalgain       Capital gain                                 Quantitative  Relevant
    capitalloss       Capital loss                                 Quantitative  Relevant
    hoursperweek      Number of hours worked per week              Quantitative  Relevant
    native            Native country                               Nominal       Relevant
    Salary            Salary level                                 Nominal       Relevant

'''
# Code modularity must be maintained

# Import all the required libraries and modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# imbalanced-learn pipeline is being called in rather than a scikit-learn one.
# This is because we will be using SMOTE in our pipeline.
# pip install imblearn
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
# SMOTE - Synthetic Minority Over-sampling Technique
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.metrics as skmet
import joblib
from sqlalchemy import create_engine, text
from urllib.parse import quote
from sklearn.model_selection import GridSearchCV
# Loading the data set
data1 = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/Data Science/naivebayes/Assignment/Naive Bayes/SalaryData_Train.csv", encoding = "ISO-8859-1")

# Mapping the type to numeric values 1 and 0. 
# This step is required for metric calculations in model evaluation phase.

data1.info()

data = data1.drop(columns = ["age", "educationno", "capitalgain", "capitalloss", "hoursperweek"])

data.info()

data['High_Sal'] = np.where(data.Salary == ' >50K', 1, 0)

data.drop(columns = ["Salary"], inplace = True)

###############################################################################
# MYSQL
# pip install pymysql
conn_string = ("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user = "root",  # user
                               pw = "1234",  # password
                               db = "salary_db"))  # database

db = create_engine(conn_string)
###############################################################################
# PostgreSQL
# pip install psycopg2 

# Creating engine which connect to postgreSQL
# conn_string = psycopg2.connect(database = "postgres", user = 'postgres', password = 'monish1234', host = 'localhost', port= '5432')

# conn_string = ("postgresql+psycopg2://{user}:{pw}@localhost/{db}"
#                        .format(user = "postgres", # user
#                                pw = quote("postgres"), # password
#                                db = "postgres")) # database
###############################################################################
db = create_engine(conn_string)
conn = db.connect()

data.to_sql('salary_raw', con = conn, if_exists = 'replace', index = False)

conn.autocommit = True

###############################################################################
# Select query
sql = 'SELECT * from salary_raw'
salary_data = pd.read_sql_query(text(sql), conn)

 
# Data Preprocessing - textual data

# Imbalance check
salary_data.High_Sal.value_counts()
salary_data.High_Sal.value_counts() / len(salary_data.High_Sal)  # values in percentages

# alternatively
salary_data.groupby(['High_Sal']).size()
salary_data.groupby(['High_Sal']).size() / len(salary_data.High_Sal)

# Data Split
salary_train, salary_test = train_test_split(salary_data, test_size = 0.2, stratify = salary_data[['High_Sal']], random_state = 0) # StratifiedKFold is a variation of k-fold which returns stratified folds: each set contains approximately the same percentage of samples of each target class as the complete set.

# CountVectorizer
# Convert a collection of text documents to a matrix of token counts

countvectorizer = CountVectorizer(analyzer = 'word', stop_words = 'english')

###########################
# for illustrative purposes
# s_sample = salary_train.loc[salary_train.text.str.len() < 50].sample(3, random_state = 35)
# s_sample = s_sample.iloc[:, 0:2]

# # Document Term Matrix with CountVectorizer (# returns 1D array)
# s_vec = pd.DataFrame(countvectorizer.fit_transform(s_sample.values.ravel()).\
#         toarray(), columns = countvectorizer.get_feature_names_out())

# s_vec    
###########################    

# creating a matrix of token counts for the entire text document 
def split_into_words(i):
    return [word for word in i.split(" ")]

# Get a list of columns except the excluded column
columns_to_include = [col for col in salary_data.columns if col != 'High_Sal']

# Concatenate text from all columns except the excluded column into a single series
salary_data_combined_text = ''
for col in columns_to_include:
    salary_data_combined_text += salary_data[col] + ' '
    
salary_train_combined_text = ''
for col in columns_to_include:
    salary_train_combined_text += salary_train[col] + ' '

salary_test_combined_text = ''
for col in columns_to_include:
    salary_test_combined_text += salary_test[col] + ' '


# Defining the preparation of email texts into word count matrix format - Bag of Words
salary_bow = CountVectorizer(analyzer = split_into_words).fit(salary_data_combined_text)

# Defining BOW for all messages
all_salary_matrix = salary_bow.transform(salary_data_combined_text)

# For training messages
train_salary_matrix = salary_bow.transform(salary_train_combined_text)

# For testing messages
test_salary_matrix = salary_bow.transform(salary_test_combined_text)


# We will use SMOTE technique to handle class imbalance.
# Oversampling can be a good option when we have class imbalance.
# Due to this our model will perform poorly in capturing variation in a class
# because we have too few instances of that class, relative to one or more other classes.

# SMOTE: Is an approach is to oversample (duplicating examples) the minority class
# This is a type of data augmentation for the minority class and is referred 
# to as the Synthetic Minority Oversampling Technique, or SMOTE for short.
smote = SMOTE(random_state = 0)

# Transform the dataset
X_train, y_train = smote.fit_resample(train_salary_matrix, salary_train.High_Sal)

y_train.unique()
y_train.values.sum()   # Number of '1's
y_train.size - y_train.values.sum()  # Number of '0's
# The data is now balanced

# Multinomial Naive Bayes
classifier_mb = MultinomialNB()   #vanilla model with default parameters
classifier_mb.fit(X_train, y_train)

# Evaluation on Test Data
test_pred_m = classifier_mb.predict(test_salary_matrix)

pd.crosstab(salary_test.High_Sal, test_pred_m)

# Accuracy
accuracy_test_m = np.mean(test_pred_m == salary_test.High_Sal)
accuracy_test_m

# or alternatively
skmet.accuracy_score(salary_test.High_Sal, test_pred_m) 

# Training Data accuracy
train_pred_m = classifier_mb.predict(train_salary_matrix)

pd.crosstab(salary_train.High_Sal, train_pred_m)

# Accuracy
accuracy_train_m = np.mean(train_pred_m == salary_train.High_Sal)
accuracy_train_m

skmet.accuracy_score(salary_train.High_Sal, train_pred_m) 


############################################
# Model Tuning - Hyperparameter optimization

# Multinomial Naive Bayes changing default alpha for laplace smoothing
# if alpha = 0 then no smoothing is applied and the default alpha parameter is 1
# the smoothing process mainly solves the emergence of zero probability problem in the dataset.

# formula: 
# P(w|High_Sal) = (num of High_Sal with w + alpha)/(Total num of High_Sal salary + K(alpha))
# K = total num of words in the email to be classified
param_grid = {
    'alpha': [0.01, 0.1, 0.5, 1.0,5.0 , 10.0],  # Additive smoothing parameter
    'fit_prior': [True, False],  # Whether to learn class prior probabilities or not
}
clf = MultinomialNB()

# Initialize the GridSearchCV object
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)


NB_NEW = grid_search.fit(X_train, y_train)

print(NB_NEW.best_params_)


# Evaluation on Test Data after applying laplace
test_pred_lap = grid_search.predict(test_salary_matrix)

pd.crosstab(test_pred_lap, salary_test.High_Sal)

accuracy_test_lap = np.mean(test_pred_lap == salary_test.High_Sal)
accuracy_test_lap

skmet.accuracy_score(salary_test.High_Sal, test_pred_lap) 


# Training Data accuracy
train_pred_lap = grid_search.predict(train_salary_matrix)

pd.crosstab(train_pred_lap, salary_train.High_Sal)

accuracy_train_lap = np.mean(train_pred_lap == salary_train.High_Sal)
accuracy_train_lap

skmet.accuracy_score(salary_train.High_Sal, train_pred_lap) 


# Metrics
print("accuracy: %.2f, sensitivity: %.2f, specificity: %.2f, precision: %.2f"  %
  (skmet.accuracy_score(salary_test.High_Sal.ravel(), test_pred_lap),
  skmet.recall_score(salary_test.High_Sal.ravel(), test_pred_lap),
  skmet.recall_score(salary_test.High_Sal.ravel(), test_pred_lap, pos_label = 0),
  skmet.precision_score(salary_test.High_Sal.ravel(), test_pred_lap)))

# Confusion Matrix - Heat Map
cm = skmet.confusion_matrix(salary_test.High_Sal, test_pred_lap)
cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['Not High_Sal', 'High_Sal'])
cmplot.plot()
cmplot.ax_.set(title = 'High_Sal Detection Confusion Matrix', 
               xlabel = 'Predicted Value', ylabel = 'Actual Value')




# Saving the Best Model using Pipelines

# Building the Pipeline
# Defining Pipeline
pipe1 = make_pipeline(countvectorizer, smote, grid_search)

# Fit the train data
processed = pipe1.fit(salary_train_combined_text.ravel(), salary_train.High_Sal.ravel())


# Save the trained model
joblib.dump(processed, 'processed1')


# load the saved model for predictions
model = joblib.load('processed1')

# Predictions
test_pred = model.predict(salary_test_combined_text.ravel())

# Evaluation on Test Data with Metrics
# Confusion Matrix
pd.crosstab(salary_test.High_Sal, test_pred)

# Accuracy
skmet.accuracy_score(salary_test.High_Sal, test_pred)

# Metrics
print("accuracy: %.2f, sensitivity: %.2f, specificity: %.2f, precision: %.2f"  %
  (skmet.accuracy_score(salary_test.High_Sal.ravel(), test_pred),
  skmet.recall_score(salary_test.High_Sal.ravel(), test_pred),
  skmet.recall_score(salary_test.High_Sal.ravel(), test_pred, pos_label = 0),
  skmet.precision_score(salary_test.High_Sal.ravel(), test_pred)))

# Confusion Matrix - Heat Map
cm = skmet.confusion_matrix(salary_test.High_Sal, test_pred)
cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['Not High_Sal', 'High_Sal'])
cmplot.plot()
cmplot.ax_.set(title = 'High_Sal Detection Confusion Matrix', 
               xlabel = 'Predicted Value', ylabel = 'Actual Value')


