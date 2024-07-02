from src.logger import Logger
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
import os, sys

if __name__ == '__main__':
    logger = Logger()

    try:
        # data ingestion
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()
        print(f'train_path: {train_path}\n test_path: {test_path}')
        print('Data ingestion completed.')

    except Exception as e:
        logger.log('Error occurred during data ingestion', 'ERROR')
        raise e

    try:
        # data transformation
        data_transformation = DataTransformation() 
        train_data, test_data = data_transformation.initiate_data_transformation(train_path, test_path)

        #separate features and taget variable
        X_train = train_data.drop('Good/Bad', axis=1)
        Y_train = train_data['Good/Bad']

        X_test = test_data.drop('Good/Bad', axis=1)
        Y_test = test_data['Good/Bad']

        print('Data transformation completed.')

    except Exception as e:
        logger.log('Error occurred during data transformation', 'ERROR')
        raise e

    try:
        # train model
        model_trainer = ModelTrainer()
        model_trainer.initiate_model_training(X_train, Y_train, X_test, Y_test) 
        print('Model training completed.')   

    except Exception as e:
        logger.log('Error occurred during model training', 'ERROR')
        raise e
