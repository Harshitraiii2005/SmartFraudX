import os
import sys
import joblib
from typing import Iterator, Tuple, Dict, Any

from river.linear_model import LogisticRegression as RiverLogisticRegression
from river.optim import SGD
from river.metrics import Accuracy, ClassificationReport

from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from src.entity.config_entity import ModelTrainerConfig
from src.exception import MyException
from src.logger import logging


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig, transformation_artifact: DataTransformationArtifact):
        self.config = config
        self.transformation_artifact = transformation_artifact

    def load_data(self) -> Iterator[Tuple[Dict[str, float], Any]]:
        """
        Generator yielding transformed samples.
        """
        try:
            for features, label in self.transformation_artifact.transformed_stream:
                if not isinstance(features, dict):
                    features = dict(features)
                # Ensure float values
                features = {k: float(v) for k, v in features.items()}
                yield features, label

        except Exception as e:
            raise MyException(e, sys)

    def train_model(self) -> ModelTrainerArtifact:
        """
        Trains the River Logistic Regression model.
        """
        try:
            logging.info(" Starting online model training...")

            # Initialize model and metrics
            model = RiverLogisticRegression(optimizer=SGD())
            accuracy = Accuracy()
            report = ClassificationReport()

            # Process stream
            for features, label in self.load_data():
                try:
                    y_pred = model.predict_one(features)
                    accuracy.update(label, y_pred)
                    report.update(label, y_pred)
                    model.learn_one(features, label)

                except Exception as sample_error:
                    logging.error(f"Error processing sample: {sample_error}")
                    continue

            final_accuracy = accuracy.get()
            logging.info(f" Training completed. Final Accuracy: {final_accuracy:.4f}")

            # Save trained model
            os.makedirs(self.config.model_trainer_dir, exist_ok=True)
            joblib.dump(model, self.config.model_path)
            logging.info(f" Model saved at: {self.config.model_path}")

            return ModelTrainerArtifact(
                model_path=self.config.model_path,
                pipeline_path=self.transformation_artifact.pipeline_path,
                best_model_name="RiverLogisticRegression",
                best_score=final_accuracy,
                training_metrics=str(report) 
            )

        except Exception as e:
            raise MyException(e, sys)
