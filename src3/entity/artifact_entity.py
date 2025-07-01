from dataclasses import dataclass
import numpy as np

@dataclass
class DataTransformationArtifact:
    """
    Stores train-validation split arrays after data transformation.
    """
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray



@dataclass
class MetaTrainerArtifact:
    """
    Stores metadata about the trained meta-model.
    """
    meta_model_path: str
    meta_model_name: str
    validation_score: float
