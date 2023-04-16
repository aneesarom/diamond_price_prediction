import os
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == '__main__':
    print(os.getcwd())
    # Data preparation
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    # Data Transformation
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initaite_data_transformation(train_data_path, test_data_path)
    # Model Training
    model_trainer = ModelTrainer()
    model_trainer.initate_model_training(train_arr, test_arr)
