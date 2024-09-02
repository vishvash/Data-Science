import pandas as pd
import streamlit as st 
# import numpy as np
from statsmodels.tools.tools import add_constant
from sqlalchemy import create_engine
from urllib.parse import quote
import pickle, joblib

model1 = pickle.load(open('Profit.pkl', 'rb'))
impute = joblib.load('meanimpute')
winsor = joblib.load('winsor')
minmax = joblib.load('minmax')
encoding = joblib.load('encoding')

def predict_Profit(data, user, pw, db):

    engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                           .format(user = user,  # user
                                   pw = quote(pw),  # password
                                   db = db))  # database

    clean = pd.DataFrame(impute.transform(data), columns = data.select_dtypes(exclude = ['object']).columns)
    clean1 = pd.DataFrame(winsor.transform(clean),columns = data.select_dtypes(exclude = ['object']).columns)
    clean2 = pd.DataFrame(minmax.transform(clean1),columns = data.select_dtypes(exclude = ['object']).columns)
    clean3 = pd.DataFrame(encoding.transform(data), columns = encoding.get_feature_names_out(input_features = data.columns))
    clean_data = pd.concat([clean2, clean3], axis = 1)
    # P = add_constant(clean_data)
    # clean_data1 = clean_data.drop(clean_data[['WT']], axis = 1)
   
    prediction = pd.DataFrame(model1.predict(clean_data), columns = ['Profit_pred'])
    
    final = pd.concat([prediction, data], axis = 1)
    final.to_sql('Profit_predictions', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

    return final

def main():
    

    st.title("Business Profit prediction")
    st.sidebar.title("Business Profit prediction")

    # st.radio('Type of Cab you want to Book', options=['Mini', 'Sedan', 'XL', 'Premium', 'Rental'])
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Business Profit Prediction App </h2>
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
            
    user = st.sidebar.text_input("user")
    pw = st.sidebar.text_input("password",  type="password")
    db = st.sidebar.text_input("database")
    
    result = ""
    
    if st.button("Predict"):
        result = predict_Profit(data, user, pw, db)
        #st.dataframe(result) or
        #st.table(result.style.set_properties(**{'background-color': 'white','color': 'black'}))
                           
        import seaborn as sns
        cm = sns.light_palette("blue", as_cmap = True)
        st.table(result.style.background_gradient(cmap=cm))

if __name__=='__main__':
    main()

