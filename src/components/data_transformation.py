from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from src.logger import Logger
from src.utils import save_obj
from imblearn.combine import SMOTETomek
from dataclasses import dataclass
import os, sys
import pandas as pd

@dataclass
class DataTransformationConfig:
    """
    Configuration for data transformation

    Attributes:
        preprocessor_file_path: str
            Path to the preprocessor pickle file.
        used_features: str
            Path to the features used file.

    """

    # paths for preprocessor, and used features
    preprocessor_file_path = os.path.join('../artifacts', 'preprocessor.pkl')
    used_features = os.path.join('../artifacts', 'features.pkl')

class DataTransformation:
    """
    A class to handle data transformation operations including preprocessing and resampling.

    Methods:
    --------
    __init__():
        Initializes DataTransformation class with configuration and logger.

    data_transformation_obj():
        Creates a preprocessing pipeline object with imputer and scaler.

    resample_data():
        Resamples the data using SMOTETomek to deal with class imbalance.

    initiate_data_transformation():
        Initiates data transformation including preprocessing, resampling and saving the objects.

    """

    def __init__(self):
        """
        Initializes the DataTransformation class with the configuration and logger.
        """
        self.data_transformation_config = DataTransformationConfig()
        self.logger = Logger()

    # data transformation object with imputer and scaler
    def data_transformation_obj(self):   
        """
        Creates a preprocessing pipeline object with imputer and scaler.

        Returns:
        preprocessing_pipeline: Pipeline
            A scikit-learn pipeline object that includes KNN imputer and Robust scaler.

        Raises:
        Exception
            If any errors occuring while creating the preprocessing pipeling.    
        """
        try:  
            self.logger.log('Creating preprocessing pipeline...')
            preprocessing_pipeline = Pipeline(
                steps=[
                    ('imputer', KNNImputer(n_neighbors=3)),
                    ('scaler', RobustScaler())
                ]
            )  
            self.logger.log('Preprocessing pipeline created succesfully.')
            return preprocessing_pipeline
        
        except Exception as e:
            self.logger.log('Error occurred while creating preprocessing pipeline', 'ERROR')
            raise e

    
    # resampling using SMOTETomek
    def resample_data(self, df):
        """
        Resamples the data using SMOTETomek to deal with class imbalance.

        Args:
        df: pd.DataFrame
            The dataframe containing the features and target variable.

        Returns:
        resampled_df: pd.DataFrame
            The datafrme with resampled data.

        Raises:
        Exception
            If any error occurs while resampling the data.    

        """
        try:

            self.logger.log('Starting resampling using SMOTETomek...')
            # separate features and target variable
            X, Y = df.drop('Good/Bad', axis=1), df['Good/Bad']

            # initialize SMOTETomek
            resampler = SMOTETomek(sampling_strategy='auto')
            X_resampled, Y_resampled = resampler.fit_resample(X,Y)

            # combine the resampled features and target into a dataframe
            resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
            resampled_df['Good/Bad'] = Y_resampled

            self.logger.log('Resampling completed successfully.')
            return resampled_df

        except Exception as e:
            self.logger.log('Error occured while resampling the data', 'ERROR')
            raise e    
        
    def initiate_data_transformation(self, train_data_path, test_data_path):
        """
        Initiates data transformation including preprocessing, resampling and saving the objects.

        Args:
        train_data_path: str
            The path to the training csv file.

        test_data_path: str
            The path to the test csv file.

        Returns:
        tuple
            A tuple containing the resampled training data and transformed test data.

        Raises:
        Exception
            If any error occurs during this process.        
        """

        try:
            self.logger.log('Starting data transformation...')
            # get preprocessing object
            preprocessing_obj = self.data_transformation_obj()

            # read in train data and test data
            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)

            self.logger.log('Training and test data read successfully.')

            # separate features and target
            X_train, Y_train = train_data.drop('Good/Bad', axis=1), train_data['Good/Bad']
            X_test, Y_test = test_data.drop('Good/Bad', axis=1), test_data['Good/Bad']

            # transform train data
            self.logger.log('Transforming training data...')
            train_data_trans = preprocessing_obj.fit_transform(X_train)
            train_data_trans = pd.DataFrame(train_data_trans, columns=X_train.columns)
            train_data_trans['Good/Bad'] = Y_train

            # resample train data
            train_data_resampled = self.resample_data(train_data_trans)

            # transform test_data
            self.logger.log('Transforming test data...')
            test_data_trans = preprocessing_obj.transform(X_test)
            test_data_trans = pd.DataFrame(test_data_trans, columns=X_test.columns)
            test_data_trans['Good/Bad'] = Y_test

            # save the preprocessing object
            self.logger.log('Saving proprocessing object...')
            save_obj(self.data_transformation_config.preprocessor_file_path, obj=preprocessing_obj)

            # save the used features
            self.logger.log('Saving used features...')
            save_obj(self.data_transformation_config.used_features, obj=preprocessing_obj.get_feature_names_out())
            
            self.logger.log('Data transformation completed successfully.')
            return (train_data_resampled, test_data_trans)
           

        except Exception as e:
            self.logger.log('Error occurred during data transformation', 'ERROR')
            raise e        
