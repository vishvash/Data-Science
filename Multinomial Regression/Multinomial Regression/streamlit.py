import pandas as pd
import streamlit as st 
from sqlalchemy import create_engine
import joblib,pickle

model1 = pickle.load(open('multinomial.pkl', 'rb'))
impute = joblib.load('impute')
winzor = joblib.load('winzor')
minmax = joblib.load('scale')


def predict_MPG(data, user, pw, db):

    engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
    
    #engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}") #database
    clean = pd.DataFrame(impute.transform(data), columns = data.columns)
    clean1 = pd.DataFrame(winzor.transform(clean), columns = data.columns)
    clean3 = pd.DataFrame(minmax.transform(clean1), columns = data.columns)
    

    prediction = pd.DataFrame(model1.predict(clean3), columns = ['choice'])
    
    final = pd.concat([prediction, data], axis = 1)
        
    final.to_sql('mode_test', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

    return final



def main():
    
    st.title("Commuter's Choice Prediction")
    st.sidebar.title("Commuter's Choice Prediction")

    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Commuter's Choice Prediction App </h2>
    </div>
    
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    st.text("")
    

    uploadedFile = st.sidebar.file_uploader("Choose a file" , type = ['CSV','xlsx'], accept_multiple_files = False, key = "fileUploader")
    if uploadedFile is not None :
        try:

            data=pd.read_csv(uploadedFile)
        except:
                try:
                    data = pd.read_excel(uploadedFile)
                except:      
                    data = pd.DataFrame(uploadedFile)
        
    else:
        st.sidebar.warning("Upload a csv or excel file")
    
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <p style="color:white;text-align:center;">Add DataBase Credientials </p>
    </div>
    """
    st.sidebar.markdown(html_temp, unsafe_allow_html = True)
            
    user = st.sidebar.text_input("user", "Type Here")
    pw = st.sidebar.text_input("password", "Type Here")
    db = st.sidebar.text_input("database", "Type Here")
    
    result = ""
    
    if st.button("Predict"):
        result = predict_MPG(data, user, pw, db)
                           
        import seaborn as sns
        cm = sns.light_palette("blue", as_cmap = True)
        st.table(result.style.background_gradient(cmap = cm))
                           
if __name__=='__main__':
    main()


