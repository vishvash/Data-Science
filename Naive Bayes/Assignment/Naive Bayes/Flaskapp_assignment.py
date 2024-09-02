from flask import Flask, render_template, request
import re
import pandas as pd
import numpy as np
import copy
import joblib
# import psycopg2
from sqlalchemy import create_engine
  
# creating Engine which connect to postgreSQL
conn_string = ("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user = "root",  # user
                               pw = "1234",  # password
                               db = "salary_db"))
  
db = create_engine(conn_string)
conn = db.connect()

# Load the saved model
model = joblib.load('processed1')

# Define flask
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST' :
        f = request.files['file']
        salary_data = pd.read_csv(f)
        
        salary_data1 = salary_data.drop(columns = ["age", "educationno", "capitalgain", "capitalloss", "hoursperweek"])

        
        columns_to_include = [col for col in salary_data1.columns if col != 'High_Sal']
        
        salary_test_combined_text = ''
        for col in columns_to_include:
            salary_test_combined_text += salary_data1[col] + ' '
               
        test_pred_lap = pd.DataFrame(model.predict(salary_test_combined_text))
        test_pred_lap.columns = ["sal_pred"]

        final = pd.concat([salary_data, test_pred_lap], axis = 1)
        

        final.to_sql('salary_predictions', con = conn, if_exists = 'replace', index= False)
        conn.autocommit = True
               
        return render_template("new.html", Y = final.to_html(justify = 'center'))

if __name__=='__main__':
    app.run(debug = True)
