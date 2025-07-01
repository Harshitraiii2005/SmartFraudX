import sys
from typing import Iterator, Tuple, Dict, Any
from pymongo.errors import PyMongoError

from src.configuration.mongo_db_connection import MongoDBClient
from src.constants import DATABASE_NAME
from src.exception import MyException


class CreditData:
    def __init__(self) -> None:
        """
        Initialize MongoDB client connection.
        """
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            raise MyException(e, sys)

    def stream_collection_as_dict(self, collection_name: str) -> Iterator[Tuple[Dict[str, Any], int]]:
        """
        Stream records from a MongoDB collection as (features_dict, label).
        Uses batch_size for stability. Handles timeout/cancellation gracefully.
        """
        try:
            db = self.mongo_client.database
            collection = db[collection_name]

            # Use batching to avoid timeouts on large collections
            cursor = collection.find({}, batch_size=100)

            for doc in cursor:
                # Ensure the target field exists
                if "IsFraud" not in doc:
                    continue

                # Extract target label
                y = doc["IsFraud"]

                # Prepare feature dictionary, excluding unwanted fields
                x = {
                    k: v
                    for k, v in doc.items()
                    if k not in ["_id", "IsFraud", "Time"]
                }

                yield x, y

        except PyMongoError as e:
            raise MyException(f"MongoDB operation error: {e}", sys)
        except Exception as e:
            raise MyException(e, sys)
