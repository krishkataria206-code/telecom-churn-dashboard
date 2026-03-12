import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.title("Telecom Customer Churn Prediction Dashboard")

# Load dataset
df = pd.read_excel("Telco_customer_churn.xlsx")

# Basic preprocessing
df.columns = df.columns.str.replace(" ", "")

drop_cols = [
    "CustomerID","Count","Country","State","City",
    "ZipCode","LatLong","Latitude","Longitude",
    "ChurnLabel","ChurnScore","ChurnReason"
]

df.drop(columns=drop_cols, inplace=True, errors="ignore")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Encode categorical variables
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

# Train model
X = df.drop("ChurnValue", axis=1)
y = df["ChurnValue"]

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)

st.subheader("Enter Customer Details")

tenure = st.slider("Tenure Months",0,72,12)
monthly = st.number_input("Monthly Charges",0.0,200.0,70.0)
total = st.number_input("Total Charges",0.0,10000.0,1000.0)

if st.button("Predict Churn"):

    sample = X.iloc[[0]].copy()

    sample["TenureMonths"] = tenure
    sample["MonthlyCharges"] = monthly
    sample["TotalCharges"] = total

    pred = model.predict(sample)[0]
    prob = model.predict_proba(sample)[0][1]

    st.write("Churn Probability:", round(prob,2))

    if pred == 1:
        st.error("⚠ Customer likely to churn")
    else:
        st.success("✅ Customer likely to stay")