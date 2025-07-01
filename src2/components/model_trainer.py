import os
import sys
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from src2.entity.config_entity import ModelTrainerConfig
from src2.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from src.exception import MyException
from src.logger import logging


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig, transformation_artifact: DataTransformationArtifact):
        self.config = config
        self.transformation_artifact = transformation_artifact

    def build_deep_bilstm_model(self, input_shape):
        try:
            model = Sequential([
                Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
                Dropout(0.4),
                Bidirectional(LSTM(64, return_sequences=True)),
                Dropout(0.3),
                Bidirectional(LSTM(32, return_sequences=False)),
                Dense(64, activation='relu'),
                Dense(1, activation='sigmoid')
            ])

            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            logging.info("✅ BiLSTM model compiled successfully.")
            return model
        except Exception as e:
            raise MyException(e, sys)

    def train_model(self) -> ModelTrainerArtifact:
        try:
            x_train = self.transformation_artifact.x_train
            y_train = self.transformation_artifact.y_train
            x_val = self.transformation_artifact.x_val
            y_val = self.transformation_artifact.y_val

            logging.info(f"Training data shape: {x_train.shape}, Labels shape: {y_train.shape}")

            model = self.build_deep_bilstm_model(input_shape=(x_train.shape[1], x_train.shape[2]))

            os.makedirs(self.config.model_trainer_dir, exist_ok=True)
            checkpoint_path = os.path.join(self.config.model_trainer_dir, "best_bilstm_model.h5")
            checkpoint = ModelCheckpoint(
                checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            )

            history = model.fit(
                x_train, y_train,
                epochs=20,
                batch_size=32,
                validation_data=(x_val, y_val),
                callbacks=[checkpoint],
                verbose=1
            )

            final_accuracy = max(history.history['val_accuracy'])
            logging.info(f"Training complete. Best Validation Accuracy: {final_accuracy:.4f}")

            return ModelTrainerArtifact(
                model_path=checkpoint_path,
                best_model_name="DeepBiLSTM",
                best_score=final_accuracy,
                training_metrics={"val_accuracy": final_accuracy}
            )

        except Exception as e:
            raise MyException(e, sys)
