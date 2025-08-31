import os
import sys
import dill
import pandas as pd
from src.exception import CustomException


def save_object(file_path, object):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file:
            dill.dump(object, file)
    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file:
            return dill.load(file)
    except Exception as e:
        raise CustomException(e, sys)
    

def get_custom_dataframe(age, systolic_bp, diastolic_bp, blood_sugar, body_temp, heart_rate):
    try:
        data_input = {
            "Age":[age],
            "SystolicBP": [systolic_bp],
            "DiastolicBP": [diastolic_bp],
            "BloodSugar": [blood_sugar],
            "BodyTemp": [body_temp],
            "HeartRate": [heart_rate]
        }
        return pd.DataFrame(data_input)
    
    except Exception as e:
        raise CustomException(e, sys)
    
# def get_custom_dataframe(age, systolic_bp, blood_sugar):
#     try:
#         data_input = {
#             "Age":[age],
#             "SystolicBP": [systolic_bp],
#             "BloodSugar": [blood_sugar]
#         }
#         return pd.DataFrame(data_input)
    
#     except Exception as e:
#         raise CustomException(e, sys)