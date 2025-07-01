from dataclasses import dataclass
from typing import Iterator, Tuple, Dict, Any, Optional


@dataclass
class DataIngestionArtifact:
    data_stream: Iterator[Tuple[Dict[str, Any], int]]
    feature_store_file_path: str
    train_file_path: str
    test_file_path: str
    streaming_data_generator: any 

@dataclass
class DataValidationArtifact:
    schema_file_path: str
    report_file_path: str
    report_page_file_path: str


@dataclass
class DataTransformationArtifact:
    transformed_stream: Iterator[Tuple[Dict[str, float], int]]
    pipeline_path: str  


@dataclass
class ModelTrainerArtifact:
    model_path: str
    pipeline_path: Optional[str]
    best_model_name: str
    best_score: float
    training_metrics: Dict[str, Dict[str, float]]
