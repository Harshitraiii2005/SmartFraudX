import os
from src.constants import *
from dataclasses import dataclass,field
from datetime import datetime

# Timestamp to create unique pipeline runs
TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP


training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()


@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(
        training_pipeline_config.artifact_dir,
        DATA_INGESTION_DIR_NAME
    )

    feature_store_file_path: str = os.path.join(
        data_ingestion_dir,
        DATA_INGESTION_FEATURE_STORE_DIR,
        FILE_NAME
    )

    collection_name: str = DATA_INGESTION_COLLECTION_NAME


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