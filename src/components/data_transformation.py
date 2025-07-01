import os
import sys
import pickle
from typing import Tuple, Iterator, Dict, Any

from river import compose
from river.preprocessing import StandardScaler, OneHotEncoder

from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from src.logger import logging
from src.exception import MyException



def bool_to_int_dict(d: Dict[str, Any]) -> Dict[str, int]:
    return {k: int(v) for k, v in d.items()}


class DataTransformer:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

        try:
            os.makedirs(self.config.transformed_data_dir, exist_ok=True)
        except Exception as e:
            raise MyException(e, sys)

        
        self.numeric_fields = [
            "Amount",
            "HourOfDay",
            "CustomerTenureMonths",
            "NumTransactionsLast24h",
            "AvgTransactionAmount7d"
        ]

        self.categorical_fields = [
            "Currency",
            "MerchantCategory",
            "TransactionType",
            "DayOfWeek",
            "GeoLocation",
            "DeviceType"
        ]

        self.boolean_fields = [
            "CardPresent",
            "IsInternational",
            "IsNewDevice"
        ]

       
        numeric_pipeline = compose.Select(*self.numeric_fields) | StandardScaler()
        categorical_pipeline = compose.Select(*self.categorical_fields) | OneHotEncoder()
        boolean_pipeline = compose.Select(*self.boolean_fields) | compose.FuncTransformer(bool_to_int_dict)

        
        self.pipeline = compose.TransformerUnion(
            numeric_pipeline,
            categorical_pipeline,
            boolean_pipeline
        )

    def fit_transform_stream(
        self, validated_stream: Iterator[Tuple[Dict[str, Any], int]]
    ) -> Tuple[Iterator[Tuple[Dict[str, float], int]], compose.TransformerUnion]:
        """
        Applies transformation pipeline (scaling + encoding).
        """
        try:
            logging.info("Applying transformation pipeline (numeric scaling + categorical encoding...")

            def transformed_generator():
                for features, label in validated_stream:
                    try:
                        if not isinstance(features, dict):
                            raise ValueError("Features must be a dict.")

                        transformed = self.pipeline.transform_one(features)
                        self.pipeline.learn_one(features)

                        yield transformed, label

                    except Exception as e:
                        logging.error(f"Error processing sample in transformation: {e}")
                        continue

            return transformed_generator(), self.pipeline

        except Exception as e:
            raise MyException(e, sys)

    def run_transformation(
        self, validation_artifact: DataValidationArtifact
    ) -> DataTransformationArtifact:
        """
        Executes transformation and saves pipeline.
        """
        try:
            logging.info("Starting data transformation...")

            transformed_stream, pipeline = self.fit_transform_stream(
                validation_artifact.validated_stream
            )

            
            with open(self.config.pipeline_path, "wb") as f:
                pickle.dump(pipeline, f)
                logging.info(f"Transformation pipeline saved at: {self.config.pipeline_path}")

            artifact = DataTransformationArtifact(
                transformed_stream=transformed_stream,
                pipeline_path=self.config.pipeline_path
            )

            logging.info("Data transformation completed.")
            return artifact

        except Exception as e:
            raise MyException(e, sys)
