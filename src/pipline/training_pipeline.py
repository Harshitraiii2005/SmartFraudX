import sys
from src.exception import MyException
from src.logger import logging

from src.components.data_ingestion import DataIngestion

from src.entity.config_entity import (DataIngestionConfig)

from src.entity.artifact_entity import (DataIngestionArtifact)

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()


    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Enter the data ingestion stage")
            logging.info("Getting the data from mongodb")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifcat = data_ingestion.initiate_data_ingestion()
            logging.info("got the data from the mongodb")
            logging.info("exiting the data ingestion stage and data loaded successfully")
            return data_ingestion_artifcat
        except Exception as e:
            raise MyException(e, sys) from e 

    def run_pipeline(self, ) -> None:
        try:
            data_ingestion_artifact = self.start_data_ingestion()
        except Exception as e:
            raise MyException(e, sys)        