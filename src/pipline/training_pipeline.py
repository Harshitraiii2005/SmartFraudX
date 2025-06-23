import sys
from itertools import tee, islice
from src.exception import MyException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidator

from src.entity.config_entity import DataIngestionConfig, DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info(" Entering the data ingestion stage")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(" Data ingestion completed successfully")
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e, sys)

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            logging.info(" Entering the data validation stage")
            data_validator = DataValidator(config=self.data_validation_config)

            data_validation_artifact = data_validator.run_validation(
                data_stream=data_ingestion_artifact.data_stream
            )

            logging.info(" Data validation completed successfully")
            return data_validation_artifact
        except Exception as e:
            raise MyException(e, sys)

    def run_pipeline(self) -> None:
     try:
        # Step 1: Ingest data
        data_ingestion_artifact = self.start_data_ingestion()

        # ✅ Use tee to clone the stream for debugging & validation
        stream_debug, stream_validate = tee(data_ingestion_artifact.data_stream)

        # Print first few entries without exhausting the stream
        sample = list(islice(stream_debug, 2))
        print("\n🔍 DEBUG: Peeking into data stream from ingestion...")
        print(f"🧾 First 1-2 records:\n{sample}\n")

        # Pass the non-exhausted stream to validation
        data_ingestion_artifact.data_stream = stream_validate

        # Step 2: Validate data
        data_validation_artifact = self.start_data_validation(data_ingestion_artifact)

     except Exception as e:
        raise MyException(e, sys)
