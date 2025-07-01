from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load models
with open("artifact/meta_model.pkl", "rb") as f:
    meta_model = pickle.load(f)

with open("artifact/hk/model_trainer/model.pkl", "rb") as f:
    river_model = pickle.load(f)

bilstm_model = load_model(
    "artifacts/model_trainer/best_bilstm_model.keras",
    compile=False
)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/app")
def app_page():
    return render_template("app.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    # River input with lowercase keys
    river_input = {
        "Amount": data["amount"],
        "HourOfDay": data["hour_of_day"],
        "CustomerTenureMonths": data["customer_tenure"],
        "NumTransactionsLast24h": data["num_tx_last_24h"],
        "AvgTransactionAmount7d": data["avg_tx_amount_7d"],
        "CardPresent": int(data["card_present"]),
        "IsInternational": int(data["is_international"]),
        "IsNewDevice": int(data["is_new_device"])
    }

    river_prob = river_model.predict_proba_one(river_input)
    river_prob = river_prob.get(1, 0.0)

    # BiLSTM input
    bilstm_features = np.array([
        data["amount"],
        data["hour_of_day"],
        data["customer_tenure"],
        data["num_tx_last_24h"],
        data["avg_tx_amount_7d"],
        int(data["card_present"]),
        int(data["is_international"]),
        int(data["is_new_device"])
    ], dtype=np.float32)

    bilstm_input = np.tile(bilstm_features, (30, 1)).reshape(1, 30, -1)
    bilstm_prob = float(bilstm_model.predict(bilstm_input)[0][0])

    combined_probs = np.array([[river_prob, bilstm_prob]])
    pred_label = int(meta_model.predict(combined_probs)[0])
    pred_prob = float(meta_model.predict_proba(combined_probs)[0][1])

    return jsonify({
        "Prediction": "FRAUD" if pred_label else "NON-FRAUD",
        "Confidence": pred_prob * 100,
        "RiverProbability": river_prob,
        "BiLSTMProbability": bilstm_prob,
        "MetaProbability": pred_prob
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
