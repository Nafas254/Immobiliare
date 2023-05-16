import numpy as np
import pandas as pd 
import plotly as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
import joblib
import warnings
warnings.filterwarnings('ignore')

def main():
    
    st.markdown(' **DATA SET IMMOBILIARE**')

    crim = st.number_input('inserisci crim',0,2,1)
    zn = st.number_input('inserisci zn',0,500,250)
    indus = st.number_input('inserisci indus',0,500,250)
    chas = st.number_input('inserisci chas',0,500,250)
    nox = st.number_input('inserisci nox',0,500,250)
    rm = st.number_input('inserisci rm',0,500,250)
    age = st.number_input('inserisci age',0,500,250)
    dis = st.number_input('inserisci dis',0,500,250)
    rad = st.number_input('inserisci rad',0,500,250)
    tax = st.number_input('inserisci tax',0,500,250)
    ptratio = st.number_input('inserisci ptratio',0,500,250)
    b = st.number_input('inserisci b',0,500,250)
    istat = st.number_input('inserisci istat',0,500,250)

    new_model = joblib.load('immobiliare.pkl')
    res = new_model.predict([[crim,zn,indus,chas,nox,rm,age,dis,rad,tax,ptratio,b,istat]])[0]
    st.write(res)

    uploaded_file = st.file_uploader("Choose a csv file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df=df.drop(columns=['medv'])
        st.dataframe(df)

        res= new_model.predict(df)
        st.write(res)
        df['price']=res
        st.write(df)

        def convert_df(df):
            return df.to_csv().encode('utf-8')
        csv = convert_df(df)
        st.download_button( 
            label="Download data as CSV",
            data=csv,
            file_name='sample_df.csv',
            mime='text/csv',
        )





    
if __name__ == '__main__':
    main()