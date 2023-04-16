import os
import sys
import pickle
from src.exception.exception import CustomException
from src.logger.logging import logging
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def save_object(file_path, obj):
    try:
        # If directory is not available it will create the folder
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        # To save the pickle file, "wb" string stands for "write binary" mode
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_test, y_test, model):
    try:
        # Predict the testing data
        y_test_pred = model.predict(X_test)
        test_model_score = r2_score(y_test, y_test_pred)
        return test_model_score

    except Exception as e:
        logging.info('Exception occurred during model evaluating')
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        # read the final model pickle file
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception occurred in load_object function utils')
        raise CustomException(e, sys)
