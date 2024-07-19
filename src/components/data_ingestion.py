import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

sys.path.insert(0, '../src')
from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    artifacts_folder: str = "artifacts"
    raw_data_filename: str = "data.csv"
    train_data_filename: str = "train.csv"
    test_data_filename: str = "test.csv"

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Create artifacts folder if it doesn't exist
            os.makedirs(self.ingestion_config.artifacts_folder, exist_ok=True)
            logging.info(f"Artifacts folder '{self.ingestion_config.artifacts_folder}' created or already exists")

            # Define paths
            raw_data_path = os.path.join(self.ingestion_config.artifacts_folder, self.ingestion_config.raw_data_filename)
            train_data_path = os.path.join(self.ingestion_config.artifacts_folder, self.ingestion_config.train_data_filename)
            test_data_path = os.path.join(self.ingestion_config.artifacts_folder, self.ingestion_config.test_data_filename)

            # Read your data
            df = pd.read_csv(raw_data_path)
            logging.info('Read the dataset as a dataframe')

            # Save the raw data (optional)
            df.to_csv(raw_data_path, index=False, header=True)
            logging.info(f"Raw data saved to '{raw_data_path}'")

            # Train test split
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save train and test sets
            train_set.to_csv(train_data_path, index=False, header=True)
            logging.info(f"Train data saved to '{train_data_path}'")
            test_set.to_csv(test_data_path, index=False, header=True)
            logging.info(f"Test data saved to '{test_data_path}'")

            logging.info("Ingestion of the data is completed")

            return train_data_path, test_data_path

        except PermissionError as e:
            logging.error(f"Permission error: {e}")
            raise CustomException(e, sys)
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))

    print(f"Shape of transformed train data: {train_arr.shape}")
    print(f"Shape of transformed test data: {test_arr.shape}")


