import os
from datetime import date

# ---------------------- MongoDB ----------------------
DATABASE_NAME = "Credit-Card-Data"
COLLECTION_NAME = " creditcard"  # Removed extra space
MONGODB_URL_KEY = "MONGO_URI"

# ---------------------- Pipeline ----------------------
PIPELINE_NAME = "smartfraudx_pipeline"
ARTIFACT_DIR = "artifact"

# ---------------------- Model & Preprocessing ----------------------
MODEL_FILE_NAME = "model.pkl"
PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"
TARGET_COLUMN = "Class"
CURRENT_YEAR = date.today().year

# ---------------------- File Paths ----------------------
FILE_NAME = "data.csv"  # Used if exporting stream to file
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")

# ---------------------- AWS / S3 ----------------------
AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
REGION_NAME = "ap-south-1"

MODEL_BUCKET_NAME = "modelmlopsproj"
MODEL_PUSHER_S3_KEY = "model-registry"

# ---------------------- Data Ingestion ----------------------
DATA_INGESTION_COLLECTION_NAME = COLLECTION_NAME
DATA_INGESTION_DIR_NAME = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR = "feature_store"

# ---------------------- Data Validation ----------------------
DATA_VALIDATION_DIR_NAME = "data_validation"
DATA_VALIDATION_REPORT_FILE_NAME = "report.yaml"
INVALID_RECORD_LOG_FILE = "invalid_records.csv"

# ---------------------- Data Transformation ----------------------
DATA_TRANSFORMATION_DIR_NAME = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR = "transformed_object"
SCALER_FILE_NAME = "scaler.pkl"

# ---------------------- Model Trainer ----------------------
MODEL_TRAINER_DIR_NAME = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE = 0.6
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH = os.path.join("config", "model.yaml")
MODEL_TRAINER_N_ESTIMATORS = 200
MODEL_TRAINER_MIN_SAMPLES_SPLIT = 7
MODEL_TRAINER_MIN_SAMPLES_LEAF = 6
MODEL_TRAINER_MAX_DEPTH = 10
MODEL_TRAINER_CRITERION = 'entropy'
MODEL_TRAINER_RANDOM_STATE = 101

# ---------------------- Model Evaluation ----------------------
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE = 0.02

# ---------------------- App Config ----------------------
APP_HOST = "0.0.0.0"
APP_PORT = 5000
