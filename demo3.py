from src3.components.meta_trainer import MetaModelTrainer

if __name__ == "__main__":
    trainer = MetaModelTrainer()

    # Just pass CSV path
    trainer.run_meta_training(csv_path="notebook/credit_card_fraud_dataset.csv")
