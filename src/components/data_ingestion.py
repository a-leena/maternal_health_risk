import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join("artifacts", "train.csv")
    test_data_path = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self, data_path):
        logging.info("Entered the data ingestion component.")
        try:
            # loading data
            dataset = pd.read_csv(data_path)
            train, test = train_test_split(dataset, test_size=0.20, random_state=42)
            
            # creating the artifacts directory
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # saving the train and test data
            train.to_csv(self.ingestion_config.train_data_path, index=False)
            test.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Data ingestion completed.")
            
            return (self.ingestion_config.train_data_path, 
                    self.ingestion_config.test_data_path)
        
        except Exception as e:
            raise CustomException(e, sys)            

if __name__=='__main__':
    DataIngestion(r"C:\Users\aleen\Documents\Work\Data Analyst\Portfolio\EDA_ML\maternal_health_risk\data\all_features_data.csv").initiate_data_ingestion()