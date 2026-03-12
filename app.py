import streamlit as st
import pandas as pd
import pickle

# Load trained model
model = pickle.load(open("churn_model.pkl","rb"))

st.title("Telecom Customer Churn Prediction Dashboard")

st.write("Enter customer details to predict churn risk.")

# User inputs
tenure = st.slider("Tenure Months",0,72,12)
monthly_charges = st.number_input("Monthly Charges",0,200,70)
total_charges = st.number_input("Total Charges",0,10000,1000)

contract = st.selectbox("Contract Type",["Month-to-month","One year","Two year"])

internet = st.selectbox("Internet Service",["DSL","Fiber optic","No"])

tech_support = st.selectbox("Tech Support",["Yes","No"])

# Convert inputs into dataframe
input_data = pd.DataFrame({
    "TenureMonths":[tenure],
    "MonthlyCharges":[monthly_charges],
    "TotalCharges":[total_charges],
    "Contract":[contract],
    "InternetService":[internet],
    "TechSupport":[tech_support]
})

# Encode categorical values
input_data = pd.get_dummies(input_data)

# Align columns with model training
model_features = model.feature_names_in_

input_data = input_data.reindex(columns=model_features, fill_value=0)

# Prediction
if st.button("Predict Churn"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"Customer likely to churn. Risk: {probability:.2f}")
    else:
        st.success(f"Customer likely to stay. Risk: {probability:.2f}")