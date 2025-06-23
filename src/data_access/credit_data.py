import sys
import pandas as pd
import numpy as np
from typing import Optional, Iterator, Tuple, Dict, Any
from src.configuration.mongo_db_connection import MongoDBClient
from src.constants import DATABASE_NAME
from src.exception import MyException

class CreditData:
    def __init__(self) -> None:
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            raise MyException(e, sys)

    def stream_collection_as_dict(self, collection_name: str) -> Iterator[Tuple[Dict[str, Any], int]]:
        try:
            db = self.mongo_client.database
            collection = db[collection_name]
            cursor = collection.find({}) 
            for doc in cursor:
                if "Class" not in doc:
                    continue
                y = doc["Class"]
                x = {k: v for k, v in doc.items() if k not in ["_id", "Class", "Time"]}
                yield x, y
        except Exception as e:
            raise MyException(e, sys)
