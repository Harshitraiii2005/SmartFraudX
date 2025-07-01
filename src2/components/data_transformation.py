import sys
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split

from src2.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataValidationArtifact
from src2.entity.artifact_entity import DataTransformationArtifact
from src.exception import MyException
from src.logger import logging


class DataTransformation:
    def __init__(
        self,
        config: DataTransformationConfig,
        validation_artifact: DataValidationArtifact,
        validated_stream,
        window_size=30
    ):
        self.config = config
        self.validation_artifact = validation_artifact
        self.validated_stream = validated_stream
        self.window_size = window_size

    def _collect_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collect sequences from the validated stream for BiLSTM training.
        """
        rolling_window = []
        data_window = []
        label_window = []
        total_records = 0
        expected_feature_len = None

        try:
            for record in self.validated_stream:
                total_records += 1
                record = list(record)

                if len(record) < 2:
                    raise ValueError(
                        f"Record {total_records} has insufficient length (got {len(record)})."
                    )

                label = record[-1]
                features = record[:-1]

                if expected_feature_len is None:
                    expected_feature_len = len(features)
                elif len(features) != expected_feature_len:
                    raise ValueError(
                        f"Record {total_records} has inconsistent feature length: "
                        f"expected {expected_feature_len}, got {len(features)}."
                    )

                rolling_window.append(features)

                logging.info(
                    f"Record {total_records}: features length={len(features)}, rolling_window size={len(rolling_window)}"
                )

                if len(rolling_window) == self.window_size:
                    data_window.append(np.array(rolling_window))
                    label_window.append(label)
                    rolling_window.pop(0)

            if len(data_window) == 0:
                raise ValueError(
                    f"No sequences were created. Records processed: {total_records}, "
                    f"window size: {self.window_size}. Reduce window size or ensure more data."
                )

            return np.array(data_window), np.array(label_window)

        except Exception as e:
            raise MyException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Create training and validation splits from collected sequences.
        """
        try:
            logging.info("[Data Transformation] Starting BiLSTM data sequence preparation...")
            x, y = self._collect_sequences()

            
            stratify = y if len(np.unique(y)) > 1 else None

            x_train, x_val, y_train, y_val = train_test_split(
                x, y, test_size=0.2, random_state=42, stratify=stratify
            )

            logging.info(
                f"Data split complete. "
                f"x_train shape: {x_train.shape}, x_val shape: {x_val.shape}, "
                f"y_train shape: {y_train.shape}, y_val shape: {y_val.shape}"
            )

            return DataTransformationArtifact(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val
            )

        except Exception as e:
            raise MyException(e, sys)
