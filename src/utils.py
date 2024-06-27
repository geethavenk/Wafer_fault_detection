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
    
def load_obj(file_path):
    """
    Load an object from a file using pickle.

    Args:
    file_path: str
        The path from which to load the object.

    Returns:
    Object
        The loaded object.

    Raises:
    Exception
        If any error occurs while loading the object.        
    """

    try:
        logger.log(f'Loding the object from {file_path}...')

        with open(file_path, 'rb') as obj:
            loaded_obj = pickle.load(obj)
            logger.log(f'Object loaded successfully from {file_path}')
            return loaded_obj
        
    except Exception as e:
        logger.log('Error occurred while loading the object', 'ERROR')
        raise e    

