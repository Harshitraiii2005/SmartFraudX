from dataclasses import dataclass
from typing import Iterator, Tuple, Dict, Any, Optional


# Output of data ingestion
@dataclass
class DataIngestionArtifact:
    data_stream: Iterator[Tuple[Dict[str, Any], int]]


# Output of data validation
@dataclass
class DataValidationArtifact:
    validated_stream: Iterator[Tuple[Dict[str, Any], int]]
    validation_report: Dict[str, int]
    invalid_log_file_path: str


# Output of data transformation
@dataclass
class DataTransformationArtifact:
    transformed_stream: Iterator[Tuple[Dict[str, float], int]]  # River prefers dict, not np.ndarray
    scaler_path: str


# Output of model training
@dataclass
class ModelTrainerArtifact:
    model_path: str
    scaler_path: Optional[str]  # Optional if not reused later
    best_model_name: str
    best_score: float
    scaler_path: Optional[str] = None
    training_metrics: Dict[str, Dict[str, float]]  # e.g., {'0': {'precision': 0.9, ...}, '1': {...}}
