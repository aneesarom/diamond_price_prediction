"""sys module provides functions and variables which are used to manipulate different 
parts of the Python Runtime Environment """

import sys


def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()  # get the error details by exc_tb using sys method exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename  # get the file name which causes error

    error_message = f"Error occured in python script name [{file_name}] line number [{exc_tb.tb_lineno}] error message [{str(error)}]"

    return error_message  # return error message


class CustomException(Exception):  # Inheriting the Exception class to get that abilities

    def __init__(self, error_message, error_module: sys):
        super().__init__(error_message)
        # error_message exptects Exception error, error_detail expects a sys module object
        self.error_message = error_message_detail(error_message, error_detail=error_module)

    def __str__(self):
        return self.error_message
