from dataclasses import dataclass, field
import os
from datetime import datetime

from src.constants import (
    PIPELINE_NAME,
    ARTIFACT_DIR,
    DATA_INGESTION_DIR_NAME,
    DATA_INGESTION_FEATURE_STORE_DIR,
    FILE_NAME,
    DATA_INGESTION_COLLECTION_NAME,
    DATA_VALIDATION_DIR_NAME,
    DATA_VALIDATION_REPORT_FILE_NAME,
    INVALID_RECORD_LOG_FILE,
    DATA_TRANSFORMATION_DIR_NAME,
    PIPELINE_FILE_NAME,
    MODEL_TRAINER_DIR_NAME,
    MODEL_FILE_NAME,
    SCHEMA_FILE_PATH
)

from src.utils.common import read_yaml


TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP



training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()


@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = field(init=False)
    feature_store_file_path: str = field(init=False)
    collection_name: str = DATA_INGESTION_COLLECTION_NAME

    def __post_init__(self):
        self.data_ingestion_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            DATA_INGESTION_DIR_NAME
        )
        self.feature_store_file_path = os.path.join(
            self.data_ingestion_dir,
            DATA_INGESTION_FEATURE_STORE_DIR,
            FILE_NAME
        )


@dataclass
class DataValidationConfig:
    data_validation_dir: str = field(init=False)
    report_file_path: str = field(init=False)
    invalid_log_file_path: str = field(init=False)

    def __post_init__(self):
        self.data_validation_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            DATA_VALIDATION_DIR_NAME
        )
        self.report_file_path = os.path.join(
            self.data_validation_dir,
            DATA_VALIDATION_REPORT_FILE_NAME
        )
        self.invalid_log_file_path = os.path.join(
            self.data_validation_dir,
            INVALID_RECORD_LOG_FILE
        )


@dataclass
class DataTransformationConfig:
    data_transformation_dir: str = field(init=False)
    transformed_data_dir: str = field(init=False)
    pipeline_path: str = field(init=False)  # renamed from scaler_path

    def __post_init__(self):
        
        self.data_transformation_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            DATA_TRANSFORMATION_DIR_NAME
        )

        
        self.transformed_data_dir = os.path.join(
            self.data_transformation_dir,
            "transformed_data"
        )

        
        self.pipeline_path = os.path.join(
            self.data_transformation_dir,
            PIPELINE_FILE_NAME
        )


@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = field(init=False)
    model_path: str = field(init=False)
    pipeline_path: str = field(init=False)  
    hyperparams: dict = field(init=False)

    def __post_init__(self):
        self.model_trainer_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            MODEL_TRAINER_DIR_NAME
        )
        self.model_path = os.path.join(
            self.model_trainer_dir,
            MODEL_FILE_NAME
        )
        self.pipeline_path = os.path.join(   
            training_pipeline_config.artifact_dir,
            DATA_TRANSFORMATION_DIR_NAME,
            PIPELINE_FILE_NAME
        )

        schema_config = read_yaml(SCHEMA_FILE_PATH)
        self.hyperparams = schema_config.get("model_selection", {})
