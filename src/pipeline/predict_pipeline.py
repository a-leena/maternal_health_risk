import sys
from dataclasses import dataclass
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.utils import load_object

@dataclass
class PredictionConfig:
    input_preprocessor_path = "artifacts/input_preprocessor.pkl"
    target_preprocessor_path = "artifacts/target_preprocessor.pkl"
    model_path = "artifacts/model.pkl"

class PredictPipeline:
    def __init__(self):
        self.prediction_config = PredictionConfig()
    
    def predict(self, input):
        logging.info("Entered Prediction Pipeline.")
        try:
            input_preprocessor = load_object(file_path=self.prediction_config.input_preprocessor_path)
            scaled_input = input_preprocessor.transform(input)
            logging.info("Input data is preprocessed.")

            model = load_object(file_path=self.prediction_config.model_path)
            encoded_prediction = model.predict(scaled_input)
            encoded_prediction = encoded_prediction.astype(int)
            logging.info("Encoded prediction is made by the model.")

            target_preprocessor = load_object(file_path=self.prediction_config.target_preprocessor_path)
            prediction = target_preprocessor.inverse_transform(encoded_prediction)
            print(prediction)
            logging.info("Prediction is post-processed/decoded to give the class.")

            return prediction

        except Exception as e:
            raise CustomException(e, sys)