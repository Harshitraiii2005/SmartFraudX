import sys
from src.exception import MyException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src2.components.data_transformation import DataTransformation
from src2.components.model_trainer import ModelTrainer

from src.entity.config_entity import DataIngestionConfig
from src2.entity.config_entity import DataTransformationConfig, ModelTrainerConfig


class TrainingPipeline:
    def __init__(self):
        pass

    def start_data_ingestion(self):
        try:
            logging.info("🚰 Starting data ingestion...")
            ingestion = DataIngestion(data_ingestion_config=DataIngestionConfig())
            ingestion_artifact = ingestion.initiate_data_ingestion()
            logging.info("✅ Data ingestion completed.")
            return ingestion_artifact
        except Exception as e:
            raise MyException(e, sys)

    def start_data_transformation(self, ingestion_artifact):
        try:
            logging.info("🔄 Starting data transformation...")
            transformation = DataTransformation(config=DataTransformationConfig(), ingestion_artifact=ingestion_artifact)
            transformation_artifact = transformation.initiate_data_transformation()
            logging.info("✅ Data transformation completed.")
            return transformation_artifact
        except Exception as e:
            raise MyException(e, sys)

    def start_model_trainer(self, transformation_artifact):
        try:
            logging.info("🤖 Starting model training...")
            trainer = ModelTrainer(config=ModelTrainerConfig(), transformation_artifact=transformation_artifact)
            trainer_artifact = trainer.train_model()
            logging.info("✅ Model training completed.")
            return trainer_artifact
        except Exception as e:
            raise MyException(e, sys)

    def run_pipeline(self):
        try:
            logging.info("🏁 Pipeline started.")
            ingestion_artifact = self.start_data_ingestion()
            transformation_artifact = self.start_data_transformation(ingestion_artifact)
            trainer_artifact = self.start_model_trainer(transformation_artifact)
            logging.info(f"""
🎯 Pipeline Execution Summary:
   Best Model: {trainer_artifact.best_model_name}
   Accuracy Score: {trainer_artifact.best_score:.4f}
""")
        except Exception as e:
            raise MyException(e, sys)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()
