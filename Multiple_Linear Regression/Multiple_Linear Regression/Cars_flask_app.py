import pandas as pd
from statsmodels.tools.tools import add_constant
from flask import Flask, render_template, request
from sqlalchemy import create_engine
from urllib.parse import quote
import joblib, pickle

model1 = pickle.load(open('mpg.pkl', 'rb'))
impute = joblib.load('meanimpute')
winsor = joblib.load('winsor')
minmax = joblib.load('minmax')
encoding = joblib.load('encoding')


app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")
@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST' :
        f = request.files['file']
        user = request.form['user']
        password = request.form['password']
        database = request.form['database']
        data = pd.read_csv(f)
        engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                               .format(user = user,  # user
                                       pw = quote(password),  # password
                                       db = database))  # database
        clean = pd.DataFrame(impute.transform(data), columns = data.select_dtypes(exclude = ['object']).columns)
        clean1 = pd.DataFrame(winsor.transform(clean),columns = data.select_dtypes(exclude = ['object']).columns)
        clean2 = pd.DataFrame(minmax.transform(clean1),columns = data.select_dtypes(exclude = ['object']).columns)
        clean3 = pd.DataFrame(encoding.transform(data).todense(), columns = encoding.get_feature_names_out(input_features = data.columns))
        clean_data = pd.concat([clean2, clean3], axis = 1)
        P = add_constant(clean_data)
        clean_data1 = clean_data.drop(clean_data[['WT']], axis = 1)
       
        prediction = pd.DataFrame(model1.predict(clean_data1), columns = ['MPG_pred'])
        
        final = pd.concat([prediction, data], axis = 1)
        final.to_sql('mpg_predictions', con = engine, if_exists = 'replace', chunksize = 1000, index= False)
        return render_template("data.html", Z = "Your results are here", Y = final.to_html(justify='center').replace('<table border="1" class="dataframe">','<table border="1" class="dataframe" bordercolor="#000000" bgcolor="#FFCC66">'))

if __name__ == '__main__':

    app.run(debug = True)
