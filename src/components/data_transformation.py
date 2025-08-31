import os
import sys
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

@dataclass
class DataTransformationConfig:
    input_preprocessor_object_path = os.path.join("artifacts", "input_preprocessor.pkl")
    target_preprocessor_object_path = os.path.join("artifacts", "target_preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
    
    def get_preprocessor(self, numerical_cols, categorical_cols):
        try:
            numerical_pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ("label_encoder", LabelEncoder())
                ]
            )

            input_preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline", numerical_pipeline, numerical_cols)
                ]
            )
            target_preprocessor = ColumnTransformer(
                [
                    ("categorical_pipeline", categorical_pipeline, categorical_cols)
                ]
            )

            logging.info("Preprocessing pipelines have been created.")
            
            return input_preprocessor, target_preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        logging.info("Entered the data transformation component.")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            numerical_cols = train_df.select_dtypes(exclude="category").columns
            categorical_cols = train_df.select_dtypes(include="category").columns

            input_preprocessor, target_preprocessor = self.get_preprocessor(numerical_cols=numerical_cols, categorical_cols=categorical_cols)

            input_train = train_df.drop(columns=[categorical_cols])
            target_train = train_df[categorical_cols]

            input_test = test_df.drop(columns=[categorical_cols])
            target_test = test_df[categorical_cols]

            input_train_array = input_preprocessor.fit_transform(input_train)
            input_test_array = input_preprocessor.transform(input_test)

            target_train_array = target_preprocessor.fit_transform(target_train)
            target_test_array = target_preprocessor.transform(target_test)

            train_array = np.c_[input_train_array, target_train_array]
            test_array = np.c_[input_test_array, target_test_array]

            save_object(file_path=self.transformation_config.input_preprocessor_object_path, object=input_preprocessor)
            save_object(file_path=self.transformation_config.target_preprocessor_object_path, object=target_preprocessor)
            logging.info("Saving preprocessor objects.")
            
            return (train_array, test_array)
        
        except Exception as e:
            raise CustomException(e, sys)
    