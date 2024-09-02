from flask import Flask, render_template, request
import re
import pandas as pd

import pickle
import joblib


impute = joblib.load('meanimpute')
winsor = joblib.load('winsor')
minmax = joblib.load('minmax')
encoding = joblib.load('encoding')

bagging = pickle.load(open('baggingmodel.pkl', 'rb'))
rfc = pickle.load(open('rfc.pkl', 'rb'))
adaboost = pickle.load(open('adaboost.pkl', 'rb'))
gradiantboost = pickle.load(open('gradiantboostparam.pkl', 'rb'))
xgboost = pickle.load(open('Randomizedsearch_xgb.pkl', 'rb'))


# Connecting to SQL by creating a sqlachemy engine
from sqlalchemy import create_engine
from urllib.parse import quote
# engine = create_engine("mssql://@{server}/{database}?driver={driver}"
#                             .format(server = "LAPTOP-PUUHHRN1\SQLEXPRESS",               # Server name
#                                   database = "movies_db",                                # Database
#                                   driver = "ODBC Driver 17 for SQL Server")) 

# creating engine to connect database
user = 'root' # user name
pw = '1234' # password
db = 'movies_db' # database

engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")


app = Flask(__name__)

# Define flask

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
        engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                               .format(user = user,  # user
                                       pw = quote(password),  # password
                                       db = database))  # database
        data = pd.read_excel(f)
        clean = pd.DataFrame(impute.transform(data), columns = data.select_dtypes(exclude = ['object']).columns)
        clean1 = pd.DataFrame(winsor.transform(clean), columns = data.select_dtypes(exclude = ['object']).columns)
        clean2 = pd.DataFrame(minmax.transform(clean1), columns = data.select_dtypes(exclude = ['object']).columns)
        clean3 = pd.DataFrame(encoding.transform(data), columns = encoding.get_feature_names_out())
        clean_data = pd.concat([clean2,clean3], axis = 1)
        prediction = pd.DataFrame(bagging.predict(clean_data), columns = ['Bagging_Oscar'])
        prediction1 = pd.DataFrame(rfc.predict(clean_data), columns = ['RFC_Oscar'])
        prediction2 = pd.DataFrame(adaboost.predict(clean_data), columns = ['Adaboost_Oscar'])
        prediction3 = pd.DataFrame(gradiantboost.predict(clean_data), columns = ['Gradientboost_Oscar'])
        prediction4 = pd.DataFrame(xgboost.predict(clean_data), columns = ['XGboost_Oscar'])
      
        final_data = pd.concat([prediction, prediction1, prediction2, prediction3, prediction4, data], axis = 1)
        
        final_data.to_sql('ensemble_results', con = engine, if_exists = 'replace', chunksize = 1000, index= False)
        
        return render_template("new.html", Y = final_data.to_html(justify='center').replace('<table border="1" class="dataframe">','<table border="1" class="dataframe" bordercolor="#000000" bgcolor="#FFCC66">'))

if __name__=='__main__':
    app.run(debug = False)
