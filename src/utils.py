import pickle
import pandas as pd
from src.logger import Logger
import os, sys


logger = Logger()
def save_obj(file_path, obj):

    """
    Save an object to a file using pickle.

    Args:
    file_path: str
        The path where the object should be saved.
    obj: object
        The object that should be saved.   

    Raises:
    Exception:
        If there is an error while saving the object.     
    """

    try:
        # extract dir path from a file path
        dir_path = os.path.dirname(file_path)

        #make a directory
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
            logger.log(f'Object saved successfully to {file_path}', 'INFO')

    except Exception as e:
        logger.log(f'Error occurred while saving the object to {file_path}', 'ERROR')
        raise e    