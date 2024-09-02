import pandas as pd
import numpy as np
import streamlit as st 
from sqlalchemy import create_engine
import joblib, pickle


model1=pickle.load(open('logistic.pkl', 'rb'))
impute=joblib.load('impute')
winzor=joblib.load('winzor')
minmax=joblib.load('scale')


def predict_MPG(data, user, pw, db):

    engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
    data = data.drop('CASENUM', axis = 1)
    clean = pd.DataFrame(impute.transform(data),columns=data.columns).convert_dtypes()
    clean1 = pd.DataFrame(winzor.transform(clean),columns=data.columns).convert_dtypes()
    clean3 = pd.DataFrame(minmax.transform(clean1),columns=data.columns)
    
    prediction = model1.predict(clean3)
    
    # Manually update the Best Cutoff value here
    optimal_threshold = 0.6027403450992425
    data["ATTORNEY"] = np.zeros(len(prediction))

    # taking threshold value and above the prob value will be treated as correct value 
    data.loc[prediction > optimal_threshold, "ATTORNEY"] = 1
    data[['ATTORNEY']] = data[['ATTORNEY']].astype('int64')
    data.to_sql('attorney_predictions', con = engine.connect(), if_exists = 'replace', chunksize = 1000, index = False)

    return data

def main():

    st.title("Attorney for Claims Cases prediction")
    st.sidebar.title("Attorney for Claims Cases prediction")

    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Attorney for Claims Cases Prediction App </h2>
    </div>
    
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    st.text("")
    
    uploadedFile = st.sidebar.file_uploader("Choose a file", type = ['CSV', 'xlsx'], accept_multiple_files = False, key = "fileUploader")
    if uploadedFile is not None :
        try:
            data=pd.read_csv(uploadedFile)
        except:
                try:
                    data = pd.read_excel(uploadedFile)
                except:      
                    data = pd.DataFrame(uploadedFile)
        
        
    else:
        st.sidebar.warning("Upload the new data using CSV or Excel file.")
    
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <p style="color:white;text-align:center;">Add DataBase Credientials </p>
    </div>
    """
    st.sidebar.markdown(html_temp, unsafe_allow_html = True)
            
    user = st.sidebar.text_input("user", "Type Here")
    pw = st.sidebar.text_input("password", type="password")
    db = st.sidebar.text_input("database", "Type Here")
    
    result = ""
    
    if st.button("Predict"):
        result = predict_MPG(data, user, pw, db)
                                  
        import seaborn as sns
        cm = sns.light_palette("blue", as_cmap = True)
        st.table(result.style.background_gradient(cmap = cm))

if __name__=='__main__':
    main()


