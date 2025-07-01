# src3/entity/config_entity.py

from dataclasses import dataclass

@dataclass
class MetaTrainerConfig:
    """
    Configuration paths for meta-model training and component models.
    """
    bilstm_model_path: str = "artifacts/model_trainer/best_bilstm_model.h5"
    river_model_path: str = "artifacts/hk/model_trainer/model.pkl"
    meta_model_path: str = "artifacts/meta_model/meta_model.pkl"
