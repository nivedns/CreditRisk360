import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Load model and sample data
model = joblib.load('models/xgb_credit_model.pkl')
data = pd.read_csv('data/clean_application.csv')
X = data.drop('TARGET', axis=1)
X = X.select_dtypes(include=['number', 'bool'])
X = X.astype('float64')  # For SHAP

st.set_page_config(page_title="Credit Risk Predictor", layout="centered")
st.title("💳 Credit Risk 360")
st.write("Enter customer info to assess default risk and understand contributing factors.")

# Sidebar form inputs (use actual features)
features = {}

features['AMT_CREDIT'] = st.slider('Credit Amount', 50000, 1500000, 250000)
features['AMT_GOODS_PRICE'] = st.slider('Goods Price', 50000, 1000000, 200000)
features['EXT_SOURCE_2'] = st.slider('EXT_SOURCE_2 (credit score proxy)', 0.0, 1.0, 0.5)
features['EXT_SOURCE_3'] = st.slider('EXT_SOURCE_3', 0.0, 1.0, 0.5)
features['DAYS_EMPLOYED'] = st.slider('Days Employed (negative = currently working)', -20000, 0, -1000)
features['DAYS_BIRTH'] = st.slider('Days Since Birth', -25000, -7000, -12000)
features['CODE_GENDER_M'] = st.selectbox('Gender', ['Male', 'Female']) == 'Male'
features['FLAG_OWN_CAR_Y'] = st.selectbox('Owns Car?', ['Yes', 'No']) == 'Yes'

input_df = pd.DataFrame([features])
input_df = input_df.astype('float64')

# Align with model features
expected_features = X.columns
for col in expected_features:
    if col not in input_df.columns:
        input_df[col] = 0  # fill missing ones
input_df = input_df[expected_features]

# Predict
pred_prob = model.predict_proba(input_df)[0, 1]

st.subheader(f"🧮 Predicted Default Risk: {pred_prob:.2%}")

# SHAP for this prediction

with st.expander("🔍 Explain Prediction with SHAP"):
    explainer = shap.Explainer(model, X.sample(500, random_state=42))
    shap_values = explainer(input_df)

    st.write("Feature impact:")

    shap.plots.waterfall(shap_values[0], show=False)
    fig = plt.gcf()  # ✅ grab the current matplotlib figure
    st.pyplot(fig)
    plt.clf()  # 🔁 clears figure for the next use (optional)


