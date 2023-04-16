# Basic Import
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from src.exception.exception import CustomException
from src.logger.logging import logging
from src.utils.utils import save_object
from src.utils.utils import evaluate_model
from dataclasses import dataclass
import sys
import os


@dataclass  # Used when we want to create class without init function
class ModelTrainerConfig:
    # we are defining path for final model pickel file
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        # to create path object
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self, train_array, test_array):
        try:
            # split the train test data
            X_train, y_train, X_test, y_test = train_array[:, :-1], train_array[:, -1], test_array[:, :-1], test_array[:, -1]
            logging.info('Split Dependent and Independent variables from train and test data')

            models = {
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'ElasticNet': ElasticNet()
            }

            r2_accuracy = []
            trained_models_list = []

            # looping through dictionary, create model and evaluates it
            for model in list(models.values()):
                model.fit(X_train, y_train)
                r2_score = evaluate_model(X_test, y_test, model)
                r2_accuracy.append(r2_score)
                trained_models_list.append(model)

            logging.info("Model Training completed")

            # finding best model based on accuracy
            max_value = max(r2_accuracy)
            max_index = r2_accuracy.index(max_value)
            best_model = trained_models_list[max_index]
            best_model_name = list(models.keys())[max_index]
            best_model_accuracy = r2_accuracy[max_index]

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info("Best Model pickel file saved")
            print(f"{best_model_name}: {best_model_accuracy}")
            return trained_models_list, r2_accuracy

        except Exception as e:
            logging.info('Exception occurred at Model Training')
            raise CustomException(e, sys)
