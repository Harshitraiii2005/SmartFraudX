import os
import sys
import hashlib
import math
from typing import Dict, Iterator, Tuple, Any
from src.entity.artifact_entity import DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.exception import MyException
from src.logger import logging

class DataValidation:
    def __init__(self,data_validation_config, ingestion_artifact):
          self.data_validation_config = data_validation_config
          self.ingestion_artifact = ingestion_artifact


    def initiate_data_validation(self):
        """
        Runs data validation and returns a DataValidationArtifact.
        This is a placeholder function. You should implement actual validation logic here.
        """
        # Example: you might validate schema, nulls, data types, etc.
        # For now, let's create dummy artifact paths.

        schema_file_path = "artifacts/schema/schema.yaml"
        report_file_path = "artifacts/validation/report.yaml"
        report_page_file_path = "artifacts/validation/report.html"

        # Log for clarity
        print(f"[DataValidation] Validation completed. "
              f"Schema: {schema_file_path}, Report: {report_file_path}")

        # Return the artifact
        validation_artifact = DataValidationArtifact(
            schema_file_path=schema_file_path,
            report_file_path=report_file_path,
            report_page_file_path=report_page_file_path
        )

        return validation_artifact      
        

    def is_valid(self, x_y: Tuple[Dict[str, Any], int]) -> bool:
        """
        Validate a single (features, label) tuple.
        """
        try:
            x, y = x_y

            # Validate label
            if not isinstance(y, int) or y not in [0, 1]:
                return False

            # Optionally, validate fields in x
            # For example:
            if "Amount" not in x or not isinstance(x["Amount"], (int, float)):
                return False

            return True

        except Exception as e:
            raise MyException(e, sys)

    def validate_stream(
        self, data_stream: Iterator[Tuple[Dict[str, Any], int]]
    ) -> Tuple[Iterator[Tuple[Dict[str, Any], int]], Dict[str, Any]]:
        """
        Validates the incoming data stream and yields only valid records.
        Also returns a validation report.
        """
        try:
            valid_count = 0
            invalid_count = 0
            validated_data = []

            for x_y in data_stream:
                if self.is_valid(x_y):
                    validated_data.append(x_y)
                    valid_count += 1
                else:
                    invalid_count += 1

            report = {
                "valid_records": valid_count,
                "invalid_records": invalid_count,
                "total_records": valid_count + invalid_count
            }

            return iter(validated_data), report

        except Exception as e:
            raise MyException(e, sys)

    def run_validation(
        self, data_stream: Iterator[Tuple[Dict[str, Any], int]]
    ) -> Tuple[Iterator[Tuple[Dict[str, Any], int]], Dict[str, Any]]:
        """
        Entry point to perform validation.
        """
        try:
            return self.validate_stream(data_stream)
        except Exception as e:
            raise MyException(e, sys)
