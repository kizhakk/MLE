import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier
st.set_page_config(page_title='JPM Chase - Customer Churn Prediction', page_icon='🖖')
st.image("jplogo")

st.title("Bank Customers Churn Prediction")

st.sidebar.header("Input Parameters to predict Bank Customers Churn")

st.sidebar.header("Settings")
st.sidebar.markdown("---")
input_features = []
def user_input_features():
    churn_creditscore = st.sidebar.slider('Credit Score', 300, 900, 600,25)
    churn_gender = st.sidebar.selectbox('Gender', ["Male", "Female"])
    churn_gender = 1
    churn_age = st.sidebar.slider('Age', 10, 90, 40, 5)
    churn_geography = st.sidebar.selectbox('Geography', ["France", "Spain", "Germany"])
    churn_geography = 1
    churn_hascreditcard = st.sidebar.checkbox('Has Credit Card', value = False)
    churn_activemember = st.sidebar.checkbox('Is Active Member', value = False)
    churn_activemember = 1
   
    input_features.append(churn_creditscore)
    input_features.append(churn_gender)
    input_features.append(churn_age)
    input_features.append(churn_age)
    input_features.append(churn_age)
    input_features.append(churn_age)
    input_features.append(churn_age)
    input_features.append(churn_age)
    input_features.append(churn_age)
    input_features.append(churn_age)
    input_features.append(churn_age)
    input_features.append(churn_age)
    input_features.append(churn_age)
    input_features.append(churn_age)
    input_features.append(churn_age)
    input_features.append(churn_age)
    input_features.append(churn_age)
    df = np.array([input_features])
    return features

st.sidebar.markdown("---")
st.subheader('User Input parameters')
st.write(np.array([input_features]))

st.sidebar.markdown("Made by Josh/Idris/Ramu/Rajshree/Alan")

churnmodel = pickle.load(open('churnmodel-lgbm.pkl','rb'))
input_dict = {
}
nparrraymodel = [[ 45,  93,  48,  74,  29,  58,   6,  50,  81,  21,  10, 528, 430,
       220, 494, 493, 320]]
df = user_input_features()
# Add a button to trigger prediction
if st.button('Predict'):
    #Predict
    input_features_float = [float(x) for x in input_features]
    prediction = churnmodel.predict(np.array([input_features_float]))
    st.subheader('Prediction Probability for Bank Customer Churn is ...')
    st.write(prediction)

