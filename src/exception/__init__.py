import sys
import logging

def error_message_detail(error: Exception, error_detail: sys) -> str:
    _, _, exc_tb = error_detail.exc_info()

    if exc_tb is None:
        file_name = "Unknown file"
        line_number = "Unknown line"
    else:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno

    error_message = f"Error occurred in python script: [{file_name}] at line number [{line_number}]: {str(error)}"
    logging.error(error_message)
    return error_message


class MyException(Exception):
    def __init__(self, error_message: str, error_detail: sys):
        detailed_message = error_message_detail(error_message, error_detail)
        super().__init__(detailed_message)
        self.error_message = detailed_message

    def __str__(self) -> str:
        return self.error_message
