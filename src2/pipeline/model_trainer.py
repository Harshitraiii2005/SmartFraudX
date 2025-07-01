import sys
from itertools import tee, islice
from typing import Dict, Any, Iterator, Tuple

from src.exception import MyException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src2.components.data_transformation import DataTransformation
from src2.components.model_trainer import ModelTrainer

from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
)
from src2.entity.config_entity import (
    DataTransformationConfig,
    ModelTrainerConfig
)
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact
)
from src2.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact
)


class TrainPipeline:
    def __init__(self):
        try:
            logging.info("Initializing pipeline configuration objects...")
            self.data_ingestion_config = DataIngestionConfig()
            self.data_validation_config = DataValidationConfig()
            self.data_transformation_config = DataTransformationConfig()
            self.model_trainer_config = ModelTrainerConfig()
        except Exception as e:
            raise MyException(e, sys)

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting data ingestion...")
            ingestion = DataIngestion(self.data_ingestion_config)
            artifact = ingestion.initiate_data_ingestion()
            logging.info("Data ingestion completed.")
            return artifact
        except Exception as e:
            raise MyException(e, sys)

    def start_data_validation(self, ingestion_artifact: DataIngestionArtifact):
        """
       Runs the data validation step.
      """
     try:
        logging.info("[Data Validation] Starting...")

        validation = DataValidation(
            data_validation_config=DataValidationConfig(),
            ingestion_artifact=ingestion_artifact
        )

        # This is the important line: it returns BOTH
        validation_artifact, validated_stream = validation.initiate_data_validation()

        logging.info("[Data Validation] Completed.")
        return validation_artifact, validated_stream

     except Exception as e:
        raise MyException(e, sys)



    def start_data_transformation(
      self, validation_artifact: DataValidationArtifact
      )  -> DataTransformationArtifact:
     try:
        logging.info("Starting data transformation...")
        transformer = DataTransformation(
            config=self.data_transformation_config,
            validation_artifact=validation_artifact
        )
        transformation_artifact = transformer.initiate_data_transformation()
        logging.info("Data transformation completed.")
        return transformation_artifact
     except Exception as e:
        raise MyException(e, sys)


    def start_model_trainer(
        self, transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
        try:
            logging.info(" Starting model training...")
            trainer = ModelTrainer(
                config=self.model_trainer_config,
                transformation_artifact=transformation_artifact
            )
            trainer_artifact = trainer.train_model()
            logging.info("Model training completed.")
            return trainer_artifact
        except Exception as e:
            raise MyException(e, sys)

    def run_pipeline(self) -> None:
        try:
            logging.info("Running end-to-end training pipeline...")

            
            data_ingestion_artifact = self.start_data_ingestion()

            
            stream_debug, stream_real = tee(data_ingestion_artifact.data_stream)
            sample = list(islice(stream_debug, 2))
            logging.info(f"Sample records:\n{sample}")
            data_ingestion_artifact.data_stream = stream_real

            
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact.data_stream
            )

           
            data_transformation_artifact = self.start_data_transformation(
                data_validation_artifact
            )

            
            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact
            )

            
            logging.info("Pipeline Execution Summary:")
            print("\nPipeline Execution Summary:")
            print(f"   Best Model: {model_trainer_artifact.best_model_name}")
            print(f"   Accuracy: {model_trainer_artifact.best_score:.4f}")

        except Exception as e:
            raise MyException(e, sys)
