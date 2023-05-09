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
    sepal_length = st.sidebar.slider('Credit Score', 300, 900, 600,25)
    sepal_width = st.sidebar.selectbox('Gender', ["Male", "Female"])
    sepal_width = 1
    petal_length = st.sidebar.slider('Age', 10, 90, 40, 5)
    churn_geography = st.sidebar.selectbox('Geography', ["France", "Spain", "Germany"])
    churn_geography = 0
    petal_width = st.sidebar.checkbox('Has Credit Card', value = False)
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

#iris = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
#X = iris.drop('species', axis=1)
#y = iris['species']

#clf = RandomForestClassifier()
#clf.fit(X, y)

churnmodel = pickle.load(open('churnmodel.pkl','rb'))
try:
    prediction_proba = churnmodel.predict_proba(df)
except Exception as e:
        print("{}: {}".format(type(e).__name__, 'Error Encountered"))
#prediction = churnmodel.predict(df)

#st.write(prediction)

#st.subheader('Class labels and their corresponding index number')
#st.write(iris.species.unique())

#st.subheader('Prediction')
#st.write(iris.species[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
