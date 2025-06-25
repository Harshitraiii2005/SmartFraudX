import numpy as np
import pickle
from tensorflow.keras.models import load_model
from river.linear_model import LogisticRegression


def pad_features(input_array, target_length=29):
    """
    Pads or truncates the input array to the required number of features.
    """
    current_length = input_array.shape[1]
    if current_length < target_length:
        padding = np.zeros((input_array.shape[0], target_length - current_length))
        return np.hstack((input_array, padding))
    elif current_length > target_length:
        return input_array[:, :target_length]
    return input_array


class HybridFraudService:
    def __init__(self, logistic_model_path: str, bilstm_model_path: str):
        with open(logistic_model_path, "rb") as f:
            self.logistic_model = pickle.load(f)  # river logistic model

        self.bilstm_model = load_model(bilstm_model_path)

    def predict(self, input_data: np.ndarray) -> dict:
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)

        # Step 1: Use River Logistic model
        feature_dict = {f"x{i}": input_data[0][i] for i in range(input_data.shape[1])}
        logistic_pred = int(round(self.logistic_model.predict_one(feature_dict)))
        fraud_prob = float(self.logistic_model.predict_proba_one(feature_dict).get(1, 0.0))

        # Step 2: Use BiLSTM Model (requires 29 features)
        input_data_padded = pad_features(input_data, target_length=29)
        bilstm_input = input_data_padded.reshape(input_data.shape[0], 1, 29)
        bilstm_score = float(self.bilstm_model.predict(bilstm_input, verbose=0)[0][0])

        # Step 3: Interpret BILSTM result
        if bilstm_score > 0.75:
            pattern_message = "⚠️ Suspicious transaction pattern detected."
        elif bilstm_score > 0.4:
            pattern_message = "⚠️ Mildly unusual pattern."
        else:
            pattern_message = "✅ Normal behavior."

        return {
            "is_fraud": logistic_pred,
            "fraud_probability": round(fraud_prob, 4),
            "pattern_score": round(bilstm_score, 4),
            "pattern_message": pattern_message
        }
