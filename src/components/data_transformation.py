import os
import sys
import pickle
from typing import Tuple, Iterator, Dict, Any

from river.preprocessing import StandardScaler as RiverScaler

from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from src.logger import logging
from src.exception import MyException


class DataTransformer:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.scaler = RiverScaler()

        try:
            os.makedirs(self.config.transformed_data_dir, exist_ok=True)
        except Exception as e:
            raise MyException(e, sys)

    def fit_transform_stream(
        self, validated_stream: Iterator[Tuple[Dict[str, Any], int]]
    ) -> Tuple[Iterator[Tuple[Dict[str, float], int]], RiverScaler]:
        """
        Applies River's online StandardScaler to the stream and returns transformed stream.
        """
        try:
            logging.info("📊 Applying River StandardScaler on stream...")

            def transformed_generator():
                for features, label in validated_stream:
                    try:
                        if not isinstance(features, dict):
                            if isinstance(features, (list, tuple)):
                                features = {f"f{i}": val for i, val in enumerate(features)}
                            else:
                                features = dict(features)

                        # First transform, then learn
                        scaled_features = self.scaler.transform_one(features)
                        scaled_features = {k: float(v) for k, v in scaled_features.items()}

                        # Important: Update scaler for next sample
                        self.scaler.learn_one(features)

                        yield scaled_features, label

                    except Exception as e:
                        logging.error(f"Error processing sample in transformation: {e}")
                        continue

            # Return generator and scaler (save later after processing)
            return transformed_generator(), self.scaler

        except Exception as e:
            raise MyException(e, sys)

    def run_transformation(
        self, validation_artifact: DataValidationArtifact
    ) -> DataTransformationArtifact:
        """
        Executes the transformation and saves the trained scaler.
        """
        try:
            logging.info("🚀 Starting data transformation...")

            transformed_stream, scaler = self.fit_transform_stream(
                validation_artifact.validated_stream
            )

            # Save scaler after transformation setup
            with open(self.config.scaler_path, "wb") as f:
                pickle.dump(scaler, f)
                logging.info(f"💾 River scaler saved at: {self.config.scaler_path}")

            artifact = DataTransformationArtifact(
                transformed_stream=transformed_stream,
                scaler_path=self.config.scaler_path
            )

            logging.info("✅ Data transformation completed.")
            return artifact

        except Exception as e:
            raise MyException(e, sys)
