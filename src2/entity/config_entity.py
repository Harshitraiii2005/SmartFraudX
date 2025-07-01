from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    sequence_length: int = 30
    num_features: int = 8  

    def validate(self):
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be > 0")
        if self.num_features <= 0:
            raise ValueError("num_features must be > 0")

@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = "artifacts/model_trainer"
