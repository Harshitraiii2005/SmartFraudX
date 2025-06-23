from dataclasses import dataclass
from typing import Iterator, Tuple, Dict, Any

@dataclass
class DataIngestionArtifact:
    data_stream: Iterator[Tuple[Dict[str, Any], int]]
