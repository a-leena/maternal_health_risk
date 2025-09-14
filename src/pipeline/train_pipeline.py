import sys
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.components import data_ingestion, data_transformation, model_trainer

@dataclass
class TrainingConfig:
    data_path = r"C:\\Users\\aleen\Documents\Work\Data Analyst\\Portfolio\\EDA_ML\\maternal_health_risk\data\\all_features_data.csv"
    # data_path = r"C:\\Users\\aleen\Documents\Work\Data Analyst\\Portfolio\\EDA_ML\\maternal_health_risk\data\\few_features_data.csv"

class TrainPipeline:
    def __init__(self):
        self.training_config = TrainingConfig()
    
    def initiate_train_pipeline(self):
        logging.info("Entered Train Pipeline.")
        try:
            data_ingestion_component = data_ingestion.DataIngestion()
            train_data_path, test_data_path = data_ingestion_component.initiate_data_ingestion(self.training_config.data_path)

            data_transformation_component = data_transformation.DataTransformation()
            train_array, test_array = data_transformation_component.initiate_data_transformation(train_data_path, test_data_path)

            model_trainer_component = model_trainer.ModelTrainer()
            best_model_name, best_test_accuracy = model_trainer_component.initiate_model_trainer(train_array, test_array)

            print(f"Best model found.\n{best_model_name} has been trained, tuned, tested.\nTest Accuracy = {best_test_accuracy}")
        
        except Exception as e:
            raise CustomException(e, sys)



if __name__ == "__main__":
    print("here")
    training_pipeline = TrainPipeline()
    training_pipeline.initiate_train_pipeline()


"""
Random Forest has been trained, tuned, tested.
Test Accuracy = 0.839572192513369
Best Params: {'bootstrap': True, 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
Best CV Score: 0.827489932885906
Test Accuracy of Best Model: 0.839572192513369
"""