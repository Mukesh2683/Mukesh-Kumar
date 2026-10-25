import streamlit as st
import joblib
import numpy as np

# Load the pre-trained model
@st.cache_resource
def load_model():
    return joblib.load('model.sav')

model = load_model()

st.title("Advertising Sales Predictor")
st.write("Enter your budget values to predict sales.")

# User inputs for features
tv_budget = st.number_input("TV Advertising Budget ($)", min_value=0.0, value=150.0, step=1.0)
radio_budget = st.number_input("Radio Advertising Budget ($)", min_value=0.0, value=25.0, step=1.0)
newspaper_budget = st.number_input("Newspaper Advertising Budget ($)", min_value=0.0, value=30.0, step=1.0)

if st.button("Predict Sales"):
    # Make prediction
    input_features = np.array([[tv_budget, radio_budget, newspaper_budget]])
    prediction = model.predict(input_features)[0]
    
    st.success(f"Predicted Sales: {prediction:.2f} units")