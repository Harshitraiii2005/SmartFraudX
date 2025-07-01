import sys
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from src.exception import MyException
from src.logger import logging
from typing import Tuple
from src3.components.meta_trainer import MetaModelTrainer

class MetaModelPipeline:
    def __init__(
        self,
        river_model_path: str = "artifact/hk/model_trainer/model.pkl",
        bilstm_model_path: str = "artifacts/model_trainer/best_bilstm_model.h5"
    ):
        """
        Initialize pipeline with paths to the trained models.
        """
        try:
            with open(river_model_path, "rb") as f:
                self.river_model = pickle.load(f)

            self.bilstm_model = load_model(bilstm_model_path)

            logging.info(
                f"Loaded River model from {river_model_path} and BiLSTM model from {bilstm_model_path}."
            )
        except Exception as e:
            raise MyException(e, sys)

    def preprocess_features(self, raw_features: dict) -> Tuple[np.ndarray, dict]:
        """
        Convert incoming raw features dictionary to a BiLSTM-ready array and River-ready dict.
        """
        try:
            bilstm_features = np.array([
                raw_features["Amount"],
                raw_features["HourOfDay"],
                raw_features["CustomerTenureMonths"],
                raw_features["NumTransactionsLast24h"],
                raw_features["AvgTransactionAmount7d"],
                int(raw_features["CardPresent"]),
                int(raw_features["IsInternational"]),
                int(raw_features["IsNewDevice"])
            ], dtype=np.float32)

            bilstm_input = np.tile(bilstm_features, (30, 1)).reshape((1, 30, 8))

            river_input = raw_features.copy()

            return bilstm_input, river_input

        except Exception as e:
            raise MyException(e, sys)

    def predict(self, raw_features: dict) -> dict:
        """
        Run predictions using both models and return combined results.
        """
        try:
            bilstm_input, river_input = self.preprocess_features(raw_features)

            bilstm_pred_prob = self.bilstm_model.predict(bilstm_input)[0][0]
            bilstm_pred_label = int(bilstm_pred_prob >= 0.5)

            river_pred_prob = self.river_model.predict_proba_one(river_input).get(1, 0.0)
            river_pred_label = int(river_pred_prob >= 0.5)

            combined_prob = (bilstm_pred_prob + river_pred_prob) / 2.0
            combined_label = int(combined_prob >= 0.5)

            logging.info(
                f"Predictions - BiLSTM prob: {bilstm_pred_prob:.4f}, River prob: {river_pred_prob:.4f}, Combined prob: {combined_prob:.4f}"
            )

            return {
                "bilstm_probability": bilstm_pred_prob,
                "bilstm_label": bilstm_pred_label,
                "river_probability": river_pred_prob,
                "river_label": river_pred_label,
                "combined_probability": combined_prob,
                "combined_label": combined_label
            }

        except Exception as e:
            raise MyException(e, sys)

    def run_pipeline(self):
        """
        Automatically run the pipeline with a sample input.
        """
        try:
            logging.info("Running pipeline with sample transaction...")

            raw_features = {
                "Amount": 95148.43,
                "HourOfDay": 19,
                "CustomerTenureMonths": 117,
                "NumTransactionsLast24h": 5,
                "AvgTransactionAmount7d": 25410.68,
                "CardPresent": True,
                "IsInternational": False,
                "IsNewDevice": False
            }

            result = self.predict(raw_features)

            print("\n--- Prediction Results ---")
            for k, v in result.items():
                print(f"{k}: {v}")
            print("--------------------------\n")

        except Exception as e:
            raise MyException(e, sys)
