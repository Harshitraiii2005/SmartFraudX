import numpy as np
import sys
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from src.exception import MyException
from src.logger import logging


class MetaModelTrainer:
    def __init__(
        self,
        river_model_path: str = "artifact/hk/model_trainer/model.pkl",
        bilstm_model_path: str = "artifacts/model_trainer/best_bilstm_model.h5"
    ):
        """
        Initialize MetaModelTrainer with pretrained River and BiLSTM models.
        """
        try:
            with open(river_model_path, "rb") as f:
                self.river_model = pickle.load(f)
            self.bilstm_model = load_model(bilstm_model_path)

            logging.info("Loaded River and BiLSTM models successfully.")
        except Exception as e:
            raise MyException(e, sys)

    def load_records_from_csv(self, csv_path: str):
        """
        Load records for meta training from a CSV file.
        Returns:
            List of (features_dict, label)
        """
        df = pd.read_csv(csv_path)

        records = []
        for _, row in df.iterrows():
            label = int(row["IsFraud"])

            features = row.drop("IsFraud").to_dict()

           
            for key, value in features.items():
                v_str = str(value).lower()
                if v_str in ("true", "false"):
                    features[key] = v_str == "true"
                elif v_str in ("0", "1"):
                    features[key] = v_str == "1"

            records.append((features, label))

        logging.info(
            f"Loaded {len(records)} records from CSV. Positive labels: {sum(label for _, label in records)}"
        )
        return records

    def create_meta_dataset(self, records, window_size=30):
        """
        Create meta dataset using predictions from River and BiLSTM.
        """
        X_meta = []
        y_meta = []
        rolling_window = []
        total_records = 0

        for raw_features, label in records:
            total_records += 1

            
            river_input = {
                "Amount": raw_features["Amount"],
                "HourOfDay": raw_features["HourOfDay"],
                "CustomerTenureMonths": raw_features["CustomerTenureMonths"],
                "NumTransactionsLast24h": raw_features["NumTransactionsLast24h"],
                "AvgTransactionAmount7d": raw_features["AvgTransactionAmount7d"],
                "CardPresent": int(raw_features["CardPresent"]),
                "IsInternational": int(raw_features["IsInternational"]),
                "IsNewDevice": int(raw_features["IsNewDevice"]),
            }

            river_prob = self.river_model.predict_proba_one(river_input)
            if isinstance(river_prob, dict):
                river_prob = river_prob.get(1, 0.0)

            bilstm_numeric = np.array([
                raw_features["Amount"],
                raw_features["HourOfDay"],
                raw_features["CustomerTenureMonths"],
                raw_features["NumTransactionsLast24h"],
                raw_features["AvgTransactionAmount7d"],
                int(raw_features["CardPresent"]),
                int(raw_features["IsInternational"]),
                int(raw_features["IsNewDevice"])
            ], dtype=np.float32)

            rolling_window.append(bilstm_numeric)
            if len(rolling_window) < window_size:
                continue

            bilstm_input = np.array(rolling_window).reshape(1, window_size, -1)
            bilstm_prob = self.bilstm_model.predict(bilstm_input)[0][0]

            X_meta.append([river_prob, bilstm_prob])
            y_meta.append(label)

            rolling_window.pop(0)

            logging.debug(
                f"Processed record {total_records}: River={river_prob:.4f}, BiLSTM={bilstm_prob:.4f}"
            )

        if not X_meta:
            raise ValueError("No meta records created. Check your data and window size.")

        logging.info(f"Meta dataset created: {len(X_meta)} samples.")
        return np.array(X_meta), np.array(y_meta)

    def train_meta_model(self, X_meta, y_meta):
        """
        Train a RandomForest classifier as the meta model.
        """
        stratify = y_meta if len(np.unique(y_meta)) > 1 else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_meta, y_meta, test_size=0.2, random_state=42, stratify=stratify
        )

        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_val)
        y_prob = clf.predict_proba(X_val)[:, 1]

        acc = accuracy_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_prob)

        logging.info(
            f"Meta model trained. Accuracy: {acc:.4f}, AUC: {auc:.4f}"
        )

        return clf

    def run_meta_training(self, records=None, csv_path=None, window_size=30):
        """
        Main method to create meta dataset and train the meta model.
        """
        try:
            logging.info("Starting meta model training...")

            if records is None and csv_path is None:
                raise ValueError("You must provide either `records` or `csv_path`.")

            if records is None:
                records = self.load_records_from_csv(csv_path)

            X_meta, y_meta = self.create_meta_dataset(records, window_size)
            logging.info(f"Meta dataset created with shape: {X_meta.shape}, labels shape: {y_meta.shape}")

            meta_model = self.train_meta_model(X_meta, y_meta)

            
            with open("artifact/meta_model.pkl", "wb") as f:
                pickle.dump(meta_model, f)

            logging.info("Meta model training complete. Model saved to artifact/meta_model.pkl")
            return meta_model

        except Exception as e:
            raise MyException(e, sys)
