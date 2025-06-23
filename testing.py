import os
from pymongo import MongoClient
import pandas as pd


mongo_uri = os.getenv("MONGODB_URL_KEY")

if not mongo_uri:
    print("❌ MONGODB_URL_KEY environment variable not found.")
else:
    try:
        client = MongoClient(mongo_uri)
        print("✅ Successfully connected to MongoDB!")

        
        print("\n🧪 Databases and Collections Available:")
        for dbname in client.list_database_names():
            print(f"📁 Database: {dbname}")
            db_obj = client[dbname]
            print(f"   📄 Collections: {db_obj.list_collection_names()}")

        
        db = client["Credit-Card-Data"]
        collection = db[" creditcard"]

        
        count = collection.count_documents({})
        print(f"\n📄 Total documents in 'creditcard': {count}")

        
        data = list(collection.find())
        
        if not data:
            print(" No documents found in 'creditcard' collection.")
        else:
            df = pd.DataFrame(data)
            print("📊 DataFrame Shape:", df.shape)
            print("📌 Columns:", df.columns.tolist())
            print("\n🔍 Preview:\n", df.head())

    except Exception as e:
        print("❌ Failed to connect or fetch data:")
        print(e)
