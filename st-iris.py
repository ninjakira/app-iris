
import pickle

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

import streamlit as st

# model
clf = pickle.load(open('iris_clf.pkl', 'rb')) # import model
target_names = np.array(['setosa', 'versicolor', 'virginica']) # target names (for display)

# sidebar (get user inputs)

st.sidebar.header("User Input Parameters")


def user_input_features():
    sepal_length = st.sidebar.slider('Sepal Length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal Width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal Length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5, 0.2)

    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}

    features = pd.DataFrame(data, index=[0])
    return features
   

df_input = user_input_features()

# prediction
prediction = clf.predict(df_input)
prediction_proba = clf.predict_proba(df_input)


# main (display results)
st.write("""
# Iris Flower Prediction App
""")


st.subheader('User Input Parameters')
st.write(df_input)


st.subheader('Class labels and their corresponding index number')
st.write(target_names)

st.subheader('Prediction')
st.write(target_names[prediction])
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
