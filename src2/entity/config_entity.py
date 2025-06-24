from dataclasses import dataclass
import os

@dataclass
class DataTransformationConfig:
    transformation_dir: str
    sequence_length: int
    num_features: int

    def __init__(self):
        self.transformation_dir = os.path.join("artifact", "data_transformation")
        self.sequence_length = 30  # Time window
        self.num_features = 29     # V1 to V28 + Amount

@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str
    model_path: str

    def __init__(self):
        self.model_trainer_dir = os.path.join("artifact", "model_trainer")
        self.model_path = os.path.join(self.model_trainer_dir, "best_bilstm_model.h5")
