import streamlit as st
import pandas as pd
import joblib

# Load the saved model, label encoders, and scaler 
model = joblib.load("model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
scaler = joblib.load("scaler.pkl")

# Feature list used during model training
feature_order = [
    'SENIORCITIZEN', 'PARTNER', 'DEPENDENTS', 'TENURE', 'PHONESERVICE',
    'MULTIPLELINES', 'INTERNETSERVICE', 'ONLINESECURITY', 'ONLINEBACKUP',
    'DEVICEPROTECTION', 'TECHSUPPORT', 'STREAMINGTV', 'STREAMINGMOVIES',
    'CONTRACT', 'PAPERLESSBILLING', 'PAYMENTMETHOD', 'MONTHLYCHARGES', 'TOTALCHARGES'
]

# Categorical columns used for label encoding
categorical_cols = [
    'PARTNER', 'DEPENDENTS', 'PHONESERVICE', 'MULTIPLELINES',
    'INTERNETSERVICE', 'ONLINESECURITY', 'ONLINEBACKUP', 'DEVICEPROTECTION',
    'TECHSUPPORT', 'STREAMINGTV', 'STREAMINGMOVIES', 'CONTRACT',
    'PAPERLESSBILLING', 'PAYMENTMETHOD'
]

# Streamlit App UI
st.title("Customer Churn Prediction")

# Input Fields
input_values = {}
gender = st.selectbox("Gender", ['Female', 'Male'])  # Not used in model
input_values['SENIORCITIZEN'] = st.selectbox("Senior Citizen", [0, 1])
input_values['PARTNER'] = st.selectbox("Partner", ['Yes', 'No'])
input_values['DEPENDENTS'] = st.selectbox("Dependents", ['Yes', 'No'])
input_values['TENURE'] = st.slider("Tenure (months)", 0, 72, 12)
input_values['PHONESERVICE'] = st.selectbox("Phone Service", ['Yes', 'No'])
input_values['MULTIPLELINES'] = st.selectbox("Multiple Lines", ['No phone service', 'No', 'Yes'])
input_values['INTERNETSERVICE'] = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
input_values['ONLINESECURITY'] = st.selectbox("Online Security", ['No', 'Yes', 'No internet service'])
input_values['ONLINEBACKUP'] = st.selectbox("Online Backup", ['No', 'Yes', 'No internet service'])
input_values['DEVICEPROTECTION'] = st.selectbox("Device Protection", ['No', 'Yes', 'No internet service'])
input_values['TECHSUPPORT'] = st.selectbox("Tech Support", ['No', 'Yes', 'No internet service'])
input_values['STREAMINGTV'] = st.selectbox("Streaming TV", ['No', 'Yes', 'No internet service'])
input_values['STREAMINGMOVIES'] = st.selectbox("Streaming Movies", ['No', 'Yes', 'No internet service'])
input_values['CONTRACT'] = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
input_values['PAPERLESSBILLING'] = st.selectbox("Paperless Billing", ['Yes', 'No'])
input_values['PAYMENTMETHOD'] = st.selectbox("Payment Method", [
    'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
])
input_values['MONTHLYCHARGES'] = st.number_input("Monthly Charges", min_value=0.0)
input_values['TOTALCHARGES'] = st.number_input("Total Charges", min_value=0.0)

# Create DataFrame
input_df = pd.DataFrame([input_values])

# Encode categorical values
for col in categorical_cols:
    le = label_encoders[col]
    input_df[col] = input_df[col].map(lambda x: x if x in le.classes_ else le.classes_[0])
    input_df[col] = le.transform(input_df[col])

# Scale numeric values
numeric_cols = ['TENURE', 'MONTHLYCHARGES', 'TOTALCHARGES']
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

# Ensure column order matches training
input_df = input_df[feature_order]

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    if prediction == 1:
        st.error(f"Prediction: Customer is likely to churn (Probability: {probability:.2%})")
    else:
        st.success(f"Prediction: Customer is likely to stay (Probability: {1 - probability:.2%})")
