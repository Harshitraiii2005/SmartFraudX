import sys
import pandas as pd
import numpy as np
from typing import Optional

from src.configuration.mongo_db_connection import MongoDBClient
from src.constants import DATABASE_NAME
from src.exception import MyException

class VehicleData:
    def __init__(self) -> None:
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            raise MyException(e, sys)

    def export_collection_as_dataframe(self, collection_name: str, database_name: Optional[str] = None) -> pd.DataFrame:
        try:
            if database_name is None:
                db = self.mongo_client.database
            else:
                db = self.mongo_client.client[database_name]

            print(f"Using database: {db.name}")
            collection = db[collection_name]
            print(f"Fetching data from collection: {collection_name}")

            docs = list(collection.find())
            print(f"Number of documents fetched: {len(docs)}")

            df = pd.DataFrame(docs)

            if "id" in df.columns.to_list():
                df = df.drop(columns=['id'], axis=1)

            df.replace({"na": np.nan}, inplace=True)

            print(f"DataFrame shape after cleaning: {df.shape}")

            return df

        except Exception as e:
            raise MyException(e, sys)