# Import libraries
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
import pickle, joblib
import numpy as np

# pip install streamlit


# Load the saved model
model = pickle.load(open('knn.pkl', 'rb'))
# ct1 = joblib.load('processed1')
ct2 = joblib.load('processed2')


def predict(data, user, pw, db):
    engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

    # data.drop(['id'], axis = 1, inplace = True) # Excluding id column
    # newprocessed1 = pd.DataFrame(ct1.transform(data), columns = data.columns)
    newprocessed2 = pd.DataFrame(ct2.transform(data), columns = data.iloc[:, :9].columns)
    predictions = pd.DataFrame(model.predict(np.array(newprocessed2)), columns = ['Type'])
    
    final = pd.concat([predictions, data.iloc[:, :9]], axis = 1)
    final.to_sql('glass_predictions', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

    return final


def main():  

    st.title("glass Prediction")
    st.sidebar.title("Glasstype Prediction: DB credentials")

    # st.radio('Type of Cab you want to Book', options=['Mini', 'Sedan', 'XL', 'Premium', 'Rental'])
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">glass Prediction Model</h2>
    </div>
    
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    st.text("")
    
    uploadedFile = st.sidebar.file_uploader("Choose a file", type = ['csv', 'xlsx'], accept_multiple_files = False, key = "fileUploader")
    if uploadedFile is not None :
        try:

            data = pd.read_csv(uploadedFile)
        except:
                try:
                    data = pd.read_excel(uploadedFile)
                except:      
                    data = pd.DataFrame(uploadedFile)
                
    else:
        st.sidebar.warning("You need to upload a csv or excel file.")
    
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <p style="color:white;text-align:center;">Enter DataBase Credientials Here:</p>
    </div>
    """
    st.sidebar.markdown(html_temp, unsafe_allow_html = True)
            
    user = st.sidebar.text_input("user", "Type Here")
    pw = st.sidebar.text_input("password", "Type Here")
    db = st.sidebar.text_input("database", "Type Here")
    
    result = ""
    
    if st.button("Predict"):
        result = predict(data, user, pw, db)
                                   
        import seaborn as sns
        cm = sns.light_palette("red", as_cmap = True)
        st.table(result.style.background_gradient(cmap = cm)) #.set_precision(2))
                           
if __name__=='__main__':
    main()


