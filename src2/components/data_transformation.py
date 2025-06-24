import sys
import numpy as np
from typing import Tuple
from src2.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src2.entity.artifact_entity import DataTransformationArtifact
from src.exception import MyException
from src.logger import logging

class DataTransformation:
    def __init__(self, config: DataTransformationConfig, ingestion_artifact: DataIngestionArtifact):
        self.config = config
        self.ingestion_artifact = ingestion_artifact

    def _collect_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        try:
            logging.info("🔄 Starting BiLSTM data sequence preparation...")

            window_size = self.config.sequence_length
            num_features = self.config.num_features

            data_window = []
            label_window = []

            rolling_window = []

            for record, label in self.ingestion_artifact.data_stream:
                features = [record.get(f"V{i}", 0.0) for i in range(1, 29)]  # V1 to V28
                features.append(record.get("Amount", 0.0))  # Add Amount

                rolling_window.append(features)

                if len(rolling_window) == window_size:
                    data_window.append(np.array(rolling_window))
                    label_window.append(label)
                    rolling_window.pop(0)  # Move window

            x_train = np.array(data_window)  # Shape: (samples, 30, 29)
            y_train = np.array(label_window)

            logging.info(f"✅ Data transformation complete. x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
            return x_train, y_train

        except Exception as e:
            raise MyException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            x_train, y_train = self._collect_sequences()

            return DataTransformationArtifact(
                x_train=x_train,
                y_train=y_train
            )

        except Exception as e:
            raise MyException(e, sys)
