import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢", layout="centered")

# Title and description
st.title("🚢 Titanic Survival Prediction")
st.markdown("Enter passenger details to predict whether they would have survived the Titanic disaster.")

# Load model and columns
model = joblib.load('logistic_model.pkl')
model_columns = joblib.load('model_columns.pkl')

# Sidebar for input
st.sidebar.header("📝 Passenger Details")
Sex = st.sidebar.radio('Sex', ['male', 'female'])
Pclass = st.sidebar.selectbox('Passenger Class (Pclass)', [1, 2, 3])
Age = st.sidebar.slider('Age', 0, 100, 25)
Fare = st.sidebar.slider('Fare (in $)', 0.0, 600.0, 50.0)

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
probability = model.predict_proba(input_data)[0][1]

# Result
st.subheader("🎯 Prediction Result")
if prediction == 1:
    st.success(f"✅ The passenger would have **Survived** (Probability: {probability:.2%})")
else:
    st.error(f"❌ The passenger would have **Not Survived** (Probability: {probability:.2%})")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")
