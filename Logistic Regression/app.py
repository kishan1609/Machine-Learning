import streamlit as st
import pandas as pd
import joblib

# Load model and columns
model = joblib.load('logistic_model.pkl')
model_columns = joblib.load('model_columns.pkl')

# User input (example using widgets)
Sex = st.selectbox('Sex', ['male', 'female'])
Pclass = st.selectbox('Pclass', [1, 2, 3])
Age = st.number_input('Age', 0, 100)
Fare = st.number_input('Fare', 0.0, 1000.0)

# Convert input into DataFrame
input_data = pd.DataFrame([{
    'Sex': 1 if Sex == 'male' else 0,
    'Pclass': Pclass,
    'Age': Age,
    'Fare': Fare
}])

# Reindex input to match training columns
input_data = input_data.reindex(columns=model_columns, fill_value=0)

# Predict
prediction = model.predict(input_data)[0]

st.write(f"Prediction: {'Survived' if prediction == 1 else 'Not Survived'}")
