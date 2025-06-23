from dataclasses import dataclass
from typing import Iterator, Tuple, Dict, Any, Optional


@dataclass
class DataIngestionArtifact:
    data_stream: Iterator[Tuple[Dict[str, Any], int]]


@dataclass
class DataValidationArtifact:
    validated_stream: Iterator[Tuple[Dict[str, Any], int]]
    validation_report: Dict[str, int]
    invalid_log_file_path: str