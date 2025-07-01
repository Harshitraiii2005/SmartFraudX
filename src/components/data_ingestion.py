import sys
from typing import Iterator, Tuple, Dict, Any

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception import MyException
from src.logger import logging
from src.data_access.credit_data import CreditData


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise MyException(e, sys)

    def stream_data_from_mongo(self) -> Iterator[Tuple[Dict[str, Any], int]]:
        """
        Streams MongoDB documents one-by-one as (x, y) tuples for River.
        """
        try:
            logging.info("Streaming data from MongoDB collection for River model")
            credit_data = CreditData()
            return credit_data.stream_collection_as_dict(self.data_ingestion_config.collection_name)

        except Exception as e:
            raise MyException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        logging.info("Initiating River-style streaming data ingestion")

        try:
            data_stream = self.stream_data_from_mongo()

            data_ingestion_artifact = DataIngestionArtifact(
                data_stream=data_stream,
                feature_store_file_path="",          
                train_file_path="",                  
                test_file_path="",                   
                streaming_data_generator=data_stream
            )

            logging.info("DataIngestionArtifact created with streaming generator")
            return data_ingestion_artifact

        except Exception as e:
            raise MyException(e, sys)
