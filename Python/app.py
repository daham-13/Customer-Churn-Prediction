import pandas as pd 
import streamlit as st 
import joblib 
import shap 
import matplotlib.pyplot as plt 
import pickle

#load models 
model = joblib.load('churn_model.pkl')
explainer = joblib.load('shap_explainer.pkl')

#app titles 
st.title("🔮 Customer Churn Prediction")
st.write("Predict churn risk and explain the decision using SHAP!")

#Input fields 
st.sidebar.header("Customer Details")
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
payment_method = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer", "Credit card"
])

#Preprocess input data 

input_data = pd.DataFrame({
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'Contract_One year': [1 if contract == "One year" else 0],
    'Contract_Two year': [1 if contract == "Two year" else 0],
    'PaymentMethod_Credit card (automatic)': [1 if payment_method == "Credit card" else 0],
    'PaymentMethod_Electronic check': [1 if payment_method == "Electronic check" else 0],
    'PaymentMethod_Mailed check': [1 if payment_method == "Mailed check" else 0]
})

# Ensure all columns are present (add missing dummy columns with 0s)
expected_columns = model.feature_names_in_
for col in expected_columns:
    if col not in input_data.columns:
        input_data[col] = 0
input_data = input_data[expected_columns]

# Predict churn risk
if st.sidebar.button("Predict"):
    # Prediction
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1]

    # Display prediction
    st.subheader("📊 Prediction")
    st.write(f"Churn Risk: **{'High' if prediction == 1 else 'Low'}** (Probability: {prediction_proba:.2%})")

    # SHAP explanation
    st.subheader("🔍 Why did the model make this prediction?")
    shap_values = explainer.shap_values(input_data)
    
    # Summary plot
    st.write("**Overall Feature Impact**")
    fig_summary, ax_summary = plt.subplots()
    shap.summary_plot(shap_values, input_data, feature_names=expected_columns, plot_type="bar", show=False)
    st.pyplot(fig_summary)

    # Force plot for this instance
    st.write("**Detailed Feature Contributions**")
    fig_force, ax_force = plt.subplots()
    shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        input_data.iloc[0],
        feature_names=expected_columns,
        matplotlib=True,
        show=False
    )
    st.pyplot(fig_force)