import pickle
import pandas as pd
from src.logger import Logger
import os, sys
import streamlit as st
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score


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
    
def evaluate_model(X_train, Y_train, X_test, Y_test, models): 
    """
    Evaluates multiple models using AUC-ROC curve.

    Args:
    X_train: pd.DataFrame
        Training features.
    Y_train: pd.Series
        Training target.
    X_test: pd.DataFrame
        Test features.
    Y_test: pd.Series
        Test target.
    models: dict
        Dictionary with model names as keys and model instances as values.

    Returns:
    dict
        A dictionary with model names as keys and their corresponding AUC-ROC score as values.

    Raises:
    Exception
        If any error occurs during evaluation process.                        
    """
    try:
        logger.log('Starting model evaluation...')

        # replace target variables for binary classification
        Y_train = Y_train.replace({-1:0, 1:1})
        Y_test = Y_test.replace({-1:0, 1:1})
        model_report = {}

        for name, model in models.items():
            logger.log(f'Training model: {name}')

            # train the model
            model.fit(X_train, Y_train)

            # predict probabilites
            Y_pred_proba = model.predict_proba(X_test)[:, 1]

            # evaluate model
            auc_score = roc_auc_score(Y_test, Y_pred_proba)
            model_report[name] = auc_score

            logger.log(f'Model: {name}, AUC-ROC score: {auc_score}')
        
        logger.log('Model evaluation completed successfully.')
        return model_report

    except Exception as e:
        logger.log('Error occurred during model evaluation.')
        raise e       

def about_me():
        st.title("About the Creator")
        c1, c2 = st.columns([1,1])
        c1.markdown("""Hey! My name is **Geetha Venkatesh**, a passionate data scientist transitioning from academia to industry.
                    This app is for detecting faults in the wafers.
                    If you have any questions or further suggestions, feel free to 
                    contact me at geetha.r.v@gmail.com""")
        c1.markdown("If you are interested :")
        c1.markdown("Github : https://github.com/geethavenk")
        c1.markdown("LinkedIn : https://www.linkedin.com/in/dr-geetha-venkatesh/")