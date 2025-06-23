import os
import sys
import yaml
from src.exception import MyException

def read_yaml(file_path: str) -> dict:
    """
    Reads a YAML file and returns the contents as a dictionary.
    
    Args:
        file_path (str): The path to the YAML file.
    
    Returns:
        dict: Parsed YAML content.
    
    Raises:
        MyException: If file is not found or YAML is invalid.
    """
    try:
        with open(file_path, 'r') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise MyException(e, sys)
