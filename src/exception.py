import sys
from src.logger import logging

def error_message_detail(error, error_detail:sys):
    _, _, exc_tb = error_detail.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    linenum = exc_tb.tb_lineno
    error_message = f"""Error occurred in Python Script\nName [{filename}]\nLine Number [{linenum}]\nError Message [{str(error)}]"""
    return error_message

class CustomException(Exception):
    def __init__(self, error, error_detail:sys):
        super().__init__(error)
        self.error_message = error_message_detail(error, error_detail)
    
    def __str__(self):
        return self.error_message
    

if __name__=="__main__":
    try:
        a=1/0
    except Exception as e:
        logging.info("Testing the custom exception class.")
        raise CustomException(e, sys)