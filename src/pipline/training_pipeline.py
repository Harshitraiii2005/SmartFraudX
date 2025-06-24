import sys
from itertools import tee, islice

from src.exception import MyException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidator
from src.components.data_transformation import DataTransformer
from src.components.model_trainer import ModelTrainer

from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)

from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact
)


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Entering the data ingestion stage")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Data ingestion completed successfully")
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e, sys)

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            logging.info("Entering the data validation stage")
            data_validator = DataValidator(config=self.data_validation_config)
            data_validation_artifact = data_validator.run_validation(
                data_stream=data_ingestion_artifact.data_stream
            )
            logging.info("Data validation completed successfully")
            return data_validation_artifact
        except Exception as e:
            raise MyException(e, sys)

    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        try:
            logging.info("Entering the data transformation stage")
            data_transformer = DataTransformer(config=self.data_transformation_config)
            data_transformation_artifact = data_transformer.run_transformation(
                validation_artifact=data_validation_artifact
            )
            logging.info("Data transformation completed successfully")
            return data_transformation_artifact
        except Exception as e:
            raise MyException(e, sys)

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            logging.info("Entering the model training stage")
            model_trainer = ModelTrainer(
                config=self.model_trainer_config,
                transformation_artifact=data_transformation_artifact
            )
            model_trainer_artifact = model_trainer.train_model()
            logging.info("Model training completed successfully")
            return model_trainer_artifact
        except Exception as e:
            raise MyException(e, sys)

    def run_pipeline(self) -> None:
        try:
            # Step 1: Ingest data
            data_ingestion_artifact = self.start_data_ingestion()

            # DEBUG: Peek at sample
            stream_debug, stream_validate = tee(data_ingestion_artifact.data_stream)
            sample = list(islice(stream_debug, 2))
            print("\n🔍 DEBUG: Peeking into data stream from ingestion...")
            print(f"🧾 First 1-2 records:\n{sample}\n")
            data_ingestion_artifact.data_stream = stream_validate

            # Step 2: Validate
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)

            # Step 3: Transform
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact)

            # Step 4: Train Model → on transformed stream
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)

            # Final output
            print(f"✅ Model Training Completed → Best Model: {model_trainer_artifact.best_model_name}")
            print(f"📈 Best Score: {model_trainer_artifact.best_score:.4f}")

        except Exception as e:
            raise MyException(e, sys)
