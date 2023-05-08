import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.title("Bank Customers Churn Prediction")
st.image("jplogo")
st.sidebar.header("Input Parameters to predict Bank Customers Churn")

def user_input_features():
    sepal_length = st.sidebar.slider('Credit Score', 300, 900, 600,25)
    sepal_width = st.sidebar.selectbox('Gender', ["Male", "Female"])
    petal_length = st.sidebar.slider('Age', 10, 90, 40, 5)
    churn_geography = st.sidebar.selectbox('Geography', ["France", "Spain", "Germany"])
    petal_width = st.sidebar.checkbox('Has Credit Card', value = False)
    churn_activemember = st.sidebar.checkbox('Is Active Member', value = False)
    data = {'Credit Score': sepal_length,
            'Gender': sepal_width,
            'Age': petal_length,
            'Geography': churn_geography,
            'Has Credit Card': petal_width,
           'Is Active Member': churn_activemember}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

iris = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
X = iris.drop('species', axis=1)
y = iris['species']

clf = RandomForestClassifier()
clf.fit(X, y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.write(prediction)

st.subheader('Class labels and their corresponding index number')
st.write(iris.species.unique())

st.subheader('Prediction')
st.write(iris.species[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
