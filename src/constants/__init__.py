import os
from datetime import date

# ---------------------- MongoDB ----------------------
DATABASE_NAME = "Credit-Card-Data"
COLLECTION_NAME = " creditcard"  # removed extra space
MONGODB_URL_KEY = "MONGO_URI"

# ---------------------- Pipeline ----------------------
PIPELINE_NAME: str = "smartfraudx_pipeline"
ARTIFACT_DIR: str = "artifact"

# ---------------------- Model & Preprocessing ----------------------
MODEL_FILE_NAME = "model.pkl"
PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"
TARGET_COLUMN = "Class"
CURRENT_YEAR = date.today().year

# ---------------------- File Names ----------------------
FILE_NAME: str = "data.csv"  # used only if exporting stream to file
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")

# ---------------------- AWS / S3 ----------------------
AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
REGION_NAME = "ap-south-1"

MODEL_BUCKET_NAME = "modelmlopsproj"
MODEL_PUSHER_S3_KEY = "model-registry"

# ---------------------- Data Ingestion ----------------------
DATA_INGESTION_COLLECTION_NAME: str = COLLECTION_NAME
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"

# ---------------------- Data Validation ----------------------
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_REPORT_FILE_NAME: str = "report.yaml"

# ---------------------- Data Transformation ----------------------
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

# ---------------------- Model Trainer ----------------------
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")
MODEL_TRAINER_N_ESTIMATORS = 200
MODEL_TRAINER_MIN_SAMPLES_SPLIT: int = 7
MODEL_TRAINER_MIN_SAMPLES_LEAF: int = 6
MIN_SAMPLES_SPLIT_MAX_DEPTH: int = 10
MIN_SAMPLES_SPLIT_CRITERION: str = 'entropy'
MIN_SAMPLES_SPLIT_RANDOM_STATE: int = 101

# ---------------------- Model Evaluation ----------------------
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02

# ---------------------- App Config ----------------------
APP_HOST = "0.0.0.0"
APP_PORT = 5000
