from dataclasses import dataclass
import numpy as np

@dataclass
class DataTransformationArtifact:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray

@dataclass
class ModelTrainerArtifact:
    model_path: str
    best_model_name: str
    best_score: float
    training_metrics: dict
