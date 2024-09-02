# Import the required libraries for Flask deployment

from flask import Flask, render_template, request
import pandas as pd
import pickle
import joblib

# Load the saved models
model = pickle.load(open('DT.pkl','rb'))
encoding = joblib.load('imp_enc_scale')
winsor = joblib.load('winsor')


# Connecting to DB (MySQL) by creating sqlachemy engine
from sqlalchemy import create_engine
from urllib.parse import quote


'''
# MS SQL Database connection

engine = create_engine("mssql://@{server}/{database}?driver={driver}"
                            .format(server = "360DIGITMG\SQLEXPRESS",        # server name
                                  database = "loandefault",                  # database
                                  driver = "ODBC Driver 17 for SQL Server")) # driver name
'''

def decision_tree(data_new):
    if "phone" in data_new.columns:
        # If it exists, drop the "phone" column
        data_new = data_new.drop(["phone"], axis = 1)
        print("Column 'phone' has been dropped.")
    else:
        # If it doesn't exist, print a message
        print("Column 'phone' does not exist.")
    clean1 = pd.DataFrame(encoding.transform(data_new), columns = encoding.get_feature_names_out())
    clean1[['num__months_loan_duration', 'num__amount',
               'num__percent_of_income',
               'num__years_at_residence',
               'num__age', 'num__existing_loans_count']] = winsor.transform(clean1[['num__months_loan_duration',
                                                                                        'num__amount', 'num__percent_of_income', 
                                                                                        'num__years_at_residence', 'num__age',
                                                                                       'num__existing_loans_count']])

    prediction = pd.DataFrame(model.predict(clean1), columns = ['default'])
    final_data = pd.concat([prediction, data_new], axis = 1)
    return(final_data)
    
            

# Define flask
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST' :
        f = request.files['file']
        user = request.form['user']
        password = request.form['password']
        database = request.form['database']
        
        data_new = pd.read_csv(f)
        engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                               .format(user = user,  # user
                                       pw = quote(password),  # password
                                       db = database))  # database

        final_data = decision_tree(data_new)
        final_data.to_sql('credit_test', con = engine, if_exists = 'replace', chunksize = 1000, index= False)
        return render_template("new.html", Y = final_data.to_html(justify = 'center'))


if __name__=='__main__':
    app.run(debug = False)
