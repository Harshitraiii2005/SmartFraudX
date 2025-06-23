import os
import sys
import hashlib
import math
from typing import Dict, Iterator, Tuple, Any
from src.entity.artifact_entity import DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.exception import MyException
from src.logger import logging


class DataValidator:
    def __init__(self, config: DataValidationConfig):
        self.config = config
        self.duplicate_cache = set()

        try:
            if self.config.invalid_log_file_path:
                os.makedirs(os.path.dirname(self.config.invalid_log_file_path), exist_ok=True)
                with open(self.config.invalid_log_file_path, "w") as f:
                    f.write("Invalid Record Log\n")
        except Exception as e:
            raise MyException(e, sys)

    def is_valid(self, x: Dict[str, Any]) -> bool:
        try:
            for v in x.values():
                if v is None:
                    return False
                if isinstance(v, float) and math.isnan(v):
                    return False
                if isinstance(v, str) and v.strip().lower() == "na":
                    return False

            record_hash = hashlib.md5(str(sorted(x.items())).encode()).hexdigest()
            if record_hash in self.duplicate_cache:
                return False

            self.duplicate_cache.add(record_hash)
            return True
        except Exception as e:
            raise MyException(e, sys)

    def validate_stream(
        self, data_stream: Iterator[Tuple[Dict[str, Any], int]]
    ) -> Tuple[Iterator[Tuple[Dict[str, Any], int]], dict]:
        try:
            cleaned = []
            skipped_count = 0

            for x, y in data_stream:
                if self.is_valid(x):
                    cleaned.append((x, y))
                else:
                    skipped_count += 1
                    if self.config.invalid_log_file_path:
                        with open(self.config.invalid_log_file_path, "a") as f:
                            f.write(f"{x}\n")

            report = {
                "valid_records": len(cleaned),
                "invalid_records_skipped": skipped_count
            }

            return iter(cleaned), report
        except Exception as e:
            raise MyException(e, sys)

    def run_validation(
        self, data_stream: Iterator[Tuple[Dict[str, Any], int]]
    ) -> DataValidationArtifact:
        try:
            logging.info(" Starting data validation")
            validated_stream, report = self.validate_stream(data_stream)
            logging.info(f" Validation completed: {report}")

            return DataValidationArtifact(
                validated_stream=validated_stream,
                validation_report=report,
                invalid_log_file_path=self.config.invalid_log_file_path
            )
        except Exception as e:
            raise MyException(e, sys)
