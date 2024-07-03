from src.logger import Logger
from src.utils import load_obj
import os, sys
import pandas as pd
from dataclasses import dataclass

@dataclass
class PredictionPipelineConfig:
    """
    Configuration for prediction pipeline

    Attributes:
        preprocessor_path: str
            Path to the preprocessor.
        model_path: str
            Path to the trained model.
        features_path: str
            Path to the features required/used.
        predictions_path: str   
            Path to the predictions.

    """
    preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
    model_path = os.path.join('artifacts', 'model.pkl')
    features_path = os.path.join('artifacts', 'features.pkl')
    predictions_path = os.path.join('predictions', 'predictions.csv')


class PredictionPipeline:
    
    def __init__(self):
        """
        Initialize the PredictionPipeline with logger and configuration.
        """
        self.logger = Logger()
        self.prediction_config = PredictionPipelineConfig()

    def predict(self, df):
        """
        Predicts outcomes based on input features data.

        Args:
        df: pd.DataFrame
            Features data for which predictions have to be made.

        Returns:
        pred: pd.DataFrame
            Predictions for the input data.    
        """

        try:

            # load preprocessor
            self.logger.log('Loading preprocessor...')
            preprocessor = load_obj(self.prediction_config.preprocessor_path)
            self.logger.log('Preprocessor loaded successfully.')

           # load model
            self.logger.log('Loading Model...')
            model = load_obj(self.prediction_config.model_path)
            self.logger.log('Model loaded successfully.')

            # load features
            self.logger.log('Loading Features...')
            features = load_obj(self.prediction_config.features_path)
            self.logger.log('Features loaded successfully.')

            # extract the required features from df
            self.logger.log('Extracting the required features from input data...')
            data_features = df[features]
            self.logger.log('Extracted the required features.')

            # preprocess data
            self.logger.log('Performing preprocessing steps...')
            data_features = preprocessor.transform(data_features)
            self.logger.log('Preprocessing completed successfully.')

            # predict
            self.logger.log('Started prediction...')
            pred = model.predict(data_features)
            pred = pd.DataFrame(pred, columns=['Predictions'])
            self.logger.log('Prediction completed successfully.')

            # save predictions
            pred.to_csv(self.prediction_config.predictions_path, index=False, header=True)
            return pred
        
        except Exception as e:
            self.logger.log('Error occurred during predition', 'ERROR')
            raise e 
