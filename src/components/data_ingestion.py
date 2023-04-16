import os
import sys
from src.logger.logging import logging
from src.exception.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass  # Used when we want to create class without init function
class DataIngestionconfig:
    # we are defining path for raw data, train data, test data
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')


class DataIngestion:
    def __init__(self):
        # to create path object
        self.ingestion_config = DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion methods Starts')
        try:
            # to read the data from the source csv file
            df = pd.read_csv(os.path.join('notebooks/data', 'gemstone.csv'))
            logging.info('Dataset read as pandas Dataframe')

            # created artifacts directory for all raw, train, test data
            # it would create empty raw.csv also but it will replace by next lines of code
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            # To save the file in the raw directory
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            # Train test split and save the .csv file to artifacts directory
            train_set, test_set = train_test_split(df, test_size=0.30, random_state=42)
            logging.info('Train test split completed')
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('Ingestion of Data is completed')

            # return test and train path as a tuple
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        except Exception as e:
            logging.info('Exception occurred at Data Ingestion stage')
            raise CustomException(e, sys)


if __name__ == "__main__":
    di_obj = DataIngestion()
    train_path, test_path = di_obj.initiate_data_ingestion()
