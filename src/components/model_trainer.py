import os
import sys
import joblib
import numpy as np
from typing import Tuple
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

from src.entity.artifact_entity import DataValidationArtifact, ModelTrainerArtifact
from src.entity.config_entity import ModelTrainerConfig
from src.exception import MyException
from src.logger import logging


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig, validation_artifact: DataValidationArtifact):
        self.config = config
        self.validation_artifact = validation_artifact

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        try:
            X, y = [], []
            for features, label in self.validation_artifact.validated_stream:
                X.append(list(features.values()))
                y.append(label)
            return np.array(X), np.array(y)
        except Exception as e:
            raise MyException(e, sys)

    def train_model(self) -> ModelTrainerArtifact:
        try:
            logging.info("Loading and preparing data...")
            X, y = self.load_data()

            logging.info("Starting Grid Search training from schema.yaml config...")
            models = {
                "LogisticRegression": {
                    "model": LogisticRegression(),
                    "params": self.config.hyperparams.get("logistic_regression", {})
                },
                "XGBoost": {
                    "model": XGBClassifier(eval_metric='logloss', use_label_encoder=False),
                    "params": self.config.hyperparams.get("xgboost", {})
                }
            }

            best_model, best_score, best_name = None, 0, ""
            for name, config in models.items():
                gs = GridSearchCV(config["model"], config["params"], cv=3, scoring='accuracy', n_jobs=-1)
                gs.fit(X, y)
                if gs.best_score_ > best_score:
                    best_model = gs.best_estimator_
                    best_score = gs.best_score_
                    best_name = name

            os.makedirs(self.config.model_trainer_dir, exist_ok=True)
            joblib.dump(best_model, self.config.model_path)
            logging.info(f"Best model ({best_name}) saved at: {self.config.model_path}")

            y_pred = best_model.predict(X)
            report = classification_report(y, y_pred, output_dict=True)

            return ModelTrainerArtifact(
                model_path=self.config.model_path,
                scaler_path=self.config.scaler_path,
                best_model_name=best_name,
                best_score=best_score,
                training_metrics=report
            )
        except Exception as e:
            raise MyException(e, sys)

