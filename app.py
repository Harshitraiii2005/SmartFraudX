import streamlit as st
import numpy as np
from hybrid_fraud_service import HybridFraudService

# Load service
predictor = HybridFraudService(
    logistic_model_path="artifact/hk/model_trainer/model.pkl",
    bilstm_model_path="artifact/model_trainer/best_bilstm_model.h5"
)

st.title("🔍 SmartFraudX - Transaction Fraud Analyzer")

# Input form
with st.form("transaction_form"):
    st.write("## Enter Transaction Details")

    input_data = []
    for i in range(7):  # assume 7 features
        val = st.number_input(f"Feature {i+1}", step=0.01, format="%.4f")
        input_data.append(val)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_array = np.array(input_data)
    result = predictor.predict(input_array)

    st.success(f"Prediction: {'FRAUD ❌' if result['is_fraud'] else 'NOT FRAUD ✅'}")
    st.metric("Fraud Probability", result["fraud_probability"])
    st.info(result["pattern_message"])
    st.caption(f"BILSTM Pattern Score: {result['pattern_score']}")
