import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Telecom Churn Dashboard", layout="centered")

st.title("Telecom Customer Churn Prediction Dashboard")
st.write("Enter customer details to predict churn risk.")

# Load dataset
df = pd.read_excel("Telco_customer_churn.xlsx")

df.columns = df.columns.str.replace(" ", "")

drop_cols = [
    "CustomerID","Count","Country","State","City",
    "ZipCode","LatLong","Latitude","Longitude",
    "ChurnLabel","ChurnScore","ChurnReason"
]

df.drop(columns=drop_cols, inplace=True, errors="ignore")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Encode categorical
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

X = df.drop("ChurnValue", axis=1)
y = df["ChurnValue"]

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)

# =========================
# INPUT FIELDS
# =========================

tenure = st.slider("Tenure Months", 0, 72, 12)

monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)

total = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month","One year","Two year"]
)

internet = st.selectbox(
    "Internet Service",
    ["DSL","Fiber optic","No"]
)

tech = st.selectbox(
    "Tech Support",
    ["Yes","No"]
)

# =========================
# PREDICTION
# =========================

if st.button("Predict Churn"):

    sample = X.iloc[[0]].copy()

    sample["TenureMonths"] = tenure
    sample["MonthlyCharges"] = monthly
    sample["TotalCharges"] = total

    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0][1]

    if prediction == 1:
        st.error(f"Customer likely to churn. Risk: {probability:.2f}")
    else:
        st.success(f"Customer likely to stay. Risk: {probability:.2f}")
