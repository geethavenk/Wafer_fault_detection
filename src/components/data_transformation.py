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
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.logger = Logger()

    # data transformation object with imputer and scaler
    def data_transformation_obj(self):   
        try:  
            preprocessing_pipeline = Pipeline(
                steps=[
                    ('imputer', KNNImputer(n_neighbors=3)),
                    ('scaler', RobustScaler())
                ]
            )  
            return preprocessing_pipeline
        
        except Exception as e:
            self.logger.log('Error occurred while creating preprocessing pipeline', 'ERROR')
            raise e

    
    # resampling using SMOTETomek
    def resample_data(self, df):
        try:

            # separate features and target variable
            X, Y = df.drop('Good/Bad', axis=1), df['Good/Bad']

            # initialize SMOTETomek
            resampler = SMOTETomek(sampling_strategy='auto')
            X_resampled, Y_resampled = resampler.fit_resample(X,Y)

            # combine the resampled features and target into a dataframe
            resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
            resampled_df['Good/Bad'] = Y_resampled

            return resampled_df

        except Exception as e:
            self.logger.log('Error occured while resampling the data', 'ERROR')
            raise e    
        
    def initiate_data_transformation(self, train_data_path, test_data_path):

        try:
            # get preprocessing object
            preprocessing_obj = self.data_transformation_obj()

            # read in train data and test data
            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)

            # separate features and target
            X_train, Y_train = train_data.drop('Good/Bad', axis=1), train_data['Good/Bad']
            X_test, Y_test = test_data.drop('Good/Bad', axis=1), test_data['Good/Bad']

            # transform train data
            train_data_trans = preprocessing_obj.fit_transform(X_train)
            train_data_trans = pd.DataFrame(train_data_trans, columns=X_train.columns)
            train_data_trans['Good/Bad'] = Y_train

            # resample train data
            train_data_resampled = self.resample_data(train_data_trans)

            # transform test_data
            test_data_trans = preprocessing_obj.transform(X_test)
            test_data_trans = pd.DataFrame(test_data_trans, columns=X_test.columns)
            test_data_trans['Good/Bad'] = Y_test

            # save the preprocessing object
            save_obj(self.data_transformation_config.preprocessor_file_path, obj=preprocessing_obj)

            # save the used features
            save_obj(self.data_transformation_config.used_features, obj=preprocessing_obj.get_feature_names_out())
            
            return (train_data_resampled, test_data_trans)
           

        except Exception as e:
            self.logger.log('Error occurred during data transformation', 'ERROR')
            raise e        
