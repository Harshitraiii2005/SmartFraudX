import os
import sys
import pickle
import numpy as np
from typing import Tuple, Iterator, Dict, Any
from sklearn.preprocessing import StandardScaler

from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from src.logger import logging
from src.exception import MyException


class DataTransformer:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.scaler = StandardScaler()

        try:
            os.makedirs(self.config.transformed_data_dir, exist_ok=True)
        except Exception as e:
            raise MyException(e, sys)

    def fit_transform_stream(
        self, validated_stream: Iterator[Tuple[Dict[str, Any], int]]
    ) -> Tuple[Iterator[Tuple[np.ndarray, int]], StandardScaler]:
        """
        Fits a StandardScaler and transforms the stream.
        """
        try:
            # Convert stream to list to fit scaler
            data_list, label_list = [], []

            for x, y in validated_stream:
                data_list.append(list(x.values()))
                label_list.append(y)

            data_array = np.array(data_list)
            labels = np.array(label_list)

            # Fit scaler
            logging.info("Fitting StandardScaler...")
            self.scaler.fit(data_array)
            transformed = self.scaler.transform(data_array)

            # Reshape if needed (e.g., for neural nets)
            transformed = transformed.reshape(transformed.shape[0], -1)

            # Save the fitted scaler
            with open(self.config.scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)
                logging.info(f"Scaler saved at: {self.config.scaler_path}")

            # Return transformed as a generator
            def generator():
                for x, y in zip(transformed, labels):
                    yield x, y

            return generator(), self.scaler

        except Exception as e:
            raise MyException(e, sys)

    def run_transformation(
        self, validation_artifact: DataValidationArtifact
    ) -> DataTransformationArtifact:
        """
        Executes transformation and returns the final DataTransformationArtifact.
        """
        try:
            logging.info("Starting data transformation...")
            transformed_stream, scaler = self.fit_transform_stream(
                validation_artifact.validated_stream
            )

            artifact = DataTransformationArtifact(
                transformed_stream=transformed_stream,
                scaler_path=self.config.scaler_path
            )
            logging.info("Data transformation completed successfully.")
            return artifact

        except Exception as e:
            raise MyException(e, sys)
