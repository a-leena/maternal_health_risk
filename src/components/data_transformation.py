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
    
    def get_preprocessor(self, numerical_cols):
        try:
            numerical_pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler())
                ]
            )
            input_preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline", numerical_pipeline, numerical_cols)
                ]
            )

            logging.info("Preprocessing pipeline has been created.")
            
            return input_preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        logging.info("Entered the data transformation component.")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            print(train_df.info())
            numerical_cols = train_df.select_dtypes(exclude="object").columns
            categorical_cols = train_df.select_dtypes(include="object").columns

            input_preprocessor = self.get_preprocessor(numerical_cols=numerical_cols)

            input_train = train_df.drop(columns=categorical_cols)
            print(input_train.sample(10))
            target_train = train_df[categorical_cols]
            print(target_train.sample(10))

            input_test = test_df.drop(columns=categorical_cols)
            print(input_test.sample(10))
            target_test = test_df[categorical_cols]
            print(target_test.sample(10))

            input_train_array = input_preprocessor.fit_transform(input_train)
            input_test_array = input_preprocessor.transform(input_test)

            target_preprocessor = LabelEncoder()
            target_train_array = target_preprocessor.fit_transform(target_train.squeeze())
            target_test_array = target_preprocessor.transform(target_test.squeeze())

            train_array = np.c_[input_train_array, target_train_array]
            test_array = np.c_[input_test_array, target_test_array]

            save_object(file_path=self.transformation_config.input_preprocessor_object_path, object=input_preprocessor)
            save_object(file_path=self.transformation_config.target_preprocessor_object_path, object=target_preprocessor)
            logging.info("Saving preprocessor objects.")
            
            return (train_array, test_array)
        
        except Exception as e:
            raise CustomException(e, sys)
    