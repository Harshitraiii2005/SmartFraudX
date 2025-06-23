from dataclasses import dataclass
from typing import Iterator, Tuple, Dict, Any
import numpy as np

# Data ingestion output
@dataclass
class DataIngestionArtifact:
    data_stream: Iterator[Tuple[Dict[str, Any], int]]

# Data validation output
@dataclass
class DataValidationArtifact:
    validated_stream: Iterator[Tuple[Dict[str, Any], int]]
    validation_report: Dict[str, int]
    invalid_log_file_path: str

# Data transformation output
@dataclass
class DataTransformationArtifact:
    transformed_stream: Iterator[Tuple[np.ndarray, int]]
    scaler_path: str  # StandardScaler path

# Model trainer output
@dataclass
class ModelTrainerArtifact:
    model_path: str
    scaler_path: str
    best_model_name: str
    best_score: float
    training_metrics: dict
