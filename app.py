import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("churn_model.pkl")

st.title("Customer Churn Prediction System")

gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", [0, 1])
dependents = st.selectbox("Dependents", [0, 1])

tenure = st.number_input("Tenure", min_value=0)

monthly_charges = st.number_input("Monthly Charges", min_value=0.0)

total_charges = st.number_input("Total Charges", min_value=0.0)

if st.button("Predict Churn"):

    input_data = pd.DataFrame([[ 
        0 if gender=="Female" else 1,
        senior,
        partner,
        dependents,
        tenure,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        monthly_charges,
        total_charges
    ]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Customer Will Churn")
    else:
        st.success("Customer Will Stay")