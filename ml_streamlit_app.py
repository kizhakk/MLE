import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier
st.image("jplogo")

st.title("Bank Customers Churn Prediction")

st.sidebar.header("Input Parameters to predict Bank Customers Churn")

st.sidebar.header("Settings")
st.sidebar.markdown("---")
def user_input_features():
    churn_creditscore = st.sidebar.slider('Credit Score', 300, 900, 600,25)
    churn_gender = st.sidebar.selectbox('Gender', ["Male", "Female"])
    churn_age = st.sidebar.slider('Age', 10, 90, 40, 5)
    churn_geography = st.sidebar.selectbox('Geography', ["France", "Spain", "Germany"])
    churn_geography = 0
    churn_hascreditcard = st.sidebar.checkbox('Has Credit Card', value = False)
    churn_activemember = st.sidebar.checkbox('Is Active Member', value = False)
    data = {'creditscore': sepal_length,
            'gender': sepal_width,
            'age': petal_length,
            'geography': churn_geography,
            'hascreditcard': petal_width,
           'isactivemember': churn_activemember}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()
st.sidebar.markdown("---")
st.subheader('User Input parameters')
st.write(df)
st.sidebar.markdown("Made by Josh/Idris/Ramu/Rajshree/Alan")

churnmodel = pickle.load(open('churnmodel-lgbm.pkl','rb'))
input_dict = {
}
nparrraymodel = [[ 45,  93,  48,  74,  29,  58,   6,  50,  81,  21,  10, 528, 430,
       220, 494, 493, 320]]

#Predict
prediction = churnmodel.predict(nparrraymodel)

st.subheader('Prediction Probability for Bank Customer Churn is ...')
st.write(prediction)
