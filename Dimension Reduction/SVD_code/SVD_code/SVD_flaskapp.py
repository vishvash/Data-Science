
# from cProfile import run
from flask import Flask, render_template, request
from sqlalchemy import create_engine
import pandas as pd

app = Flask(__name__)


import joblib
model = joblib.load("svd_DimRed")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST' :
        f = request.files['file']
        user = request.form['user']
        pw = request.form['password']
        db = request.form['databasename']
        engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
        try:

            data=pd.read_csv(f)
        except:
                try:
                    data = pd.read_excel(f)
                except:      
                    data = pd.DataFrame(f)
                  
        # Drop the unwanted features
        data1 = data.drop(["UnivID"], axis = 1)
        
        # Read only numeric data
        num_cols = data1.select_dtypes(exclude = ['object']).columns
        
        # Perform PCA using the saved model
        svd_res = pd.DataFrame(model.transform(data1[num_cols]), columns = ['svd0','svd1', 'svd2', 'svd3'])
        final = pd.concat([data.Univ, svd_res], axis = 1)

        
        final.to_sql('university_pred_svd', con = engine, if_exists = 'replace', chunksize = 1000, index = False)
        
        html_table = final.to_html(classes = 'table table-striped')
        
        return render_template("data.html", Y = f"<style>\
                    .table {{\
                        width: 50%;\
                        margin: 0 auto;\
                        border-collapse: collapse;\
                    }}\
                    .table thead {{\
                        background-color: #39648f;\
                    }}\
                    .table th, .table td {{\
                        border: 1px solid #ddd;\
                        padding: 8px;\
                        text-align: center;\
                    }}\
                        .table td {{\
                        background-color: #a8dfe3;\
                    }}\
                            .table tbody th {{\
                            background-color: #ab2c3f;\
                        }}\
                </style>\
                {html_table}")

if __name__=='__main__':
    app.run(debug = True)
