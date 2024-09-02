import pandas as pd
import streamlit as st 
# import numpy as np
import re
import pandas as pd

# Connecting to SQL by creating a sqlachemy engine
from sqlalchemy import create_engine
from urllib.parse import quote
import pickle, joblib


impute = joblib.load('meanimpute')
winsor = joblib.load('winsor')
minmax = joblib.load('minmax')
encoding = joblib.load('encoding')

bagging = pickle.load(open('baggingmodel.pkl', 'rb'))
rfc = pickle.load(open('rfc.pkl', 'rb'))
adaboost = pickle.load(open('adaboost.pkl', 'rb'))
gradiantboost = pickle.load(open('gradiantboostparam.pkl', 'rb'))
xgboost = pickle.load(open('Randomizedsearch_xgb.pkl', 'rb'))




def predict_Oscar(data, user, pw, db):

    engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                           .format(user = user,  # user
                                   pw = quote(pw),  # password
                                   db = db))  # database

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
    final_data.to_sql('ensemble_results', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

    return final_data



def main():

    st.title("Ensemble Techniques")
    st.sidebar.title("Oscar Prediction")

    # st.radio('Type of Cab you want to Book', options=['Mini', 'Sedan', 'XL', 'Premium', 'Rental'])
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Oscar Nomination Prediction </h2>
    </div>
    
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    st.text("")
    

    uploadedFile = st.sidebar.file_uploader("Choose a file", type=['csv','xlsx'], accept_multiple_files=False, key="fileUploader")
    if uploadedFile is not None :
        try:

            data = pd.read_csv(uploadedFile)
        except:
                try:
                    data = pd.read_excel(uploadedFile)
                except:      
                    data = pd.DataFrame()
        
        
    else:
        st.sidebar.warning("You need to upload a CSV or an Excel file.")
    
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <p style="color:white;text-align:center;">Add DataBase Credientials </p>
    </div>
    """
    st.sidebar.markdown(html_temp, unsafe_allow_html = True)
            
    user = st.sidebar.text_input("User")
    pw = st.sidebar.text_input("Password",  type="password")
    db = st.sidebar.text_input("Database")
    
    result = ""
    
    if st.button("Predict"):
        result = predict_Oscar(data, user, pw, db)
        #st.dataframe(result) or
        #st.table(result.style.set_properties(**{'background-color': 'white','color': 'black'}))
                           
        import seaborn as sns
        cm = sns.light_palette("blue", as_cmap = True)
        st.table(result.style.background_gradient(cmap=cm))

if __name__=='__main__':
    main()
