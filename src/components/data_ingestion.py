import os, sys
import pandas as pd
from src.logger import Logger
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    """
    Configuration for data ingestion

    Attributes:
        raw_data_path: str
            Path to the raw data csv file.
        train_data_path: str
            Path to the train data csv file.
        test_data_path: str
            Path to the test data csv file.    

    """
    # paths for raw processed data, train and test data
    data_folder = os.path.join('../data')
    processed_data_folder = os.path.join(data_folder, 'processed_data')
    raw_data_path = os.path.join(processed_data_folder, 'raw_data_processed.csv')
    train_data_path = os.path.join(processed_data_folder, 'train.csv')
    test_data_path = os.path.join(processed_data_folder, 'test.csv')

class DataIngestion:
    """
    A class to handle data ingestion operations including loading, cleaning and splitting data.

    Methods:
    --------
    __init__():
        Initializes DataIngestion class with configuration and logger.

    load_data(path_to_folder, file_format = '.csv'):
        Loads all files of a given format from the specified folder into a single dataframe

    get_cols_with_zero_std(df):
        Identifies columns with zero standard deviation.    

    get_cols_with_missing_values(df, missing_threshold=0.7):
        Identifies columns with missing values exceeding the given threshold.   

    drop_columns(df):
        Drops columns with zero standard deviation and missing values exceeding the given threshold.

    initiate_data_ingestion(path_to_files, file_format='.csv'):
        Initiates data ingestion including loading, cleaning, splitting and saving the data.         

    """

    def __init__(self):
        """
        Initializes the DataIngestion class with the configuration for data paths.
        """
        self.ingestion_config = DataIngestionConfig()
        self.logger = Logger()    

    def load_data(self, path_to_folder, file_format = '.csv'):
        """
        Loads all files of a given format from the specified folder into a single dataframe.

        Args:
        path_to_folder: str
            Path to the folder containing data files.
        file_format: str. optional (default='.csv')
            File format to filter files.

        Returns:
        pd.DataFrame
            Concatenated dataframe containing data from all files.

        Raises:
        Exception
            If any error occur during file reading or concatenations.        

        """

        try:
            self.logger.log('Starting to read all the files', 'INFO')

            # list all the files
            all_files = os.listdir(path_to_folder)

            # filter csv files
            csv_files = [file for file in all_files if file.endswith(file_format)]

            # read each csv file and store them in a list of dataframes
            dfs = [pd.read_csv(os.path.join(path_to_folder, file)) for file in csv_files] 

            # concatenate all dataframes into single dataframe
            df = pd.concat(dfs, ignore_index=True)

            self.logger.log('All the files have been loaded to a single dataframe', 'INFO')
            return df  
        
        except Exception as e:
            self.logger.log('Error occured during reading the data files', 'ERROR')
            raise e 

    def get_cols_with_zero_std(self, df):
        """
        Takes a dataframe and returns a list of column names that has zero standard deviation

        Args:
        df: pd.DataFrame
            The input dataframe for which the columns of zero standard deviation to be identified.

        Returns:
        list
            A list of column names that have zero standard deviation.   

        Raises:
        Exception
            If any error occur during identifying the columns with zero standard deviation.     

        """
        try:
            self.logger.log('Checking for columns with zero std...', 'INFO')
            cols_with_zero_std = []

            # list all the numerical columna names
            num_col_names = [col for col in df.columns if df[col].dtype != 'O']
            
            for col in num_col_names:
                #check if column std is 0
                if df[col].std() == 0: 
                    cols_with_zero_std.append(col)
            self.logger.log(f'{len(cols_with_zero_std)} columns have zero std.')         
            return cols_with_zero_std        
              

        except Exception as e:
            self.logger.log('Error in identifying columns with zero standard deviation', 'ERROR')
            raise e       

    def get_cols_with_missing_values(self, df, missing_threshold=0.7):
        """
        Identifies columns with missing values exceeding the given threshold.

        Args:
        df: pd.DataFrame
            The input dataframe to check for missing values.
        missing_threshold: float, optional (default=0.7)
            The threshold for missing values.

        Returns:
        list
            A list of column names with missing values exceeding the threshold.

        Raises:
        Exception
            If there is an error in identifying columns with missing values.    
        """
        try:
            self.logger.log(f'Checking for columsn that have missing values of more than {missing_threshold*100}%...', 'INFO')
            
            # calculate the missing value ratio for each column
            cols_missing_ratio = df.isna().sum()/df.shape[0]
            
            # list the columns with more than threshold missing values
            cols_with_missing_values = list(cols_missing_ratio[cols_missing_ratio>missing_threshold].index)
            
            self.logger.log(f'{len(cols_with_missing_values)} columns have more than {missing_threshold*100}% missing values ')
            return cols_with_missing_values
            

        except Exception as e:
            self.logger.log('Error in identifying columns with missing values exceeding the threshold', 'ERROR')
            raise e

    def drop_columns(self, df):
        """
        Drops columns with zero standard deviation and missing values exceeding the given threshold.

        Args:
        df: pd.DataFrame
            The input dataframe from which the columns have to be dropped.

        Retuns:
        pd.DataFrame
            The dataframe with specified columns dropped.

        Raises:
        Exception:
            If there is an error in dropping columns.        
        """

        try:
            # columns with zero std
            drop_columns1 = self.get_cols_with_zero_std(df)

            # columns with missing values more than a given thredhold
            drop_columns2 = self.get_cols_with_missing_values(df, missing_threshold=0.7)

            # combine columns with zero std and columns with missing values more than a given threshold
            cols_to_drop = drop_columns1+drop_columns2

            # drop the columns 
            df.drop(cols_to_drop, axis=1, inplace=True)
            
            self.logger.log('Successfully dropped the columns with zero std and missing values', 'INFO')
            return df
        
        except Exception as e:
            self.logger.log('Error in dropping the columns from dataframe', 'ERROR')
            raise e
        
    def initiate_data_ingestion(self, path_to_files, file_format='.csv'):
        """
        Initiates data ingestion including loading, cleaning, splitting and saving the data.

        Args:
        path_to_files: str
            Path to the folder contining data files.
        file_format: str, optional (default='.csv')
            File format to filter files.

        Returns:
        tuple
            A tuple containing paths to the train and test data files.

        Raises:
        Exception
            If any error occurs during data ingestion.        
        """
        
        try:
            self.logger.log('Data ingestion initialized...', 'INFO')

            # load the files from the path
            df = self.load_data(path_to_files, file_format=file_format)

            # drop the 'Unnamed: 0' column if it exists
            if 'Unnamed: 0' in df.columns:
                df.drop('Unnamed: 0', axis=1, inplace=True)

            # drop the columns with zero std and missing values
            df = self.drop_columns(df)

            # write the filtered data to raw_data_path
            # create directories based on the config file
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            # write df as csv file
            df.to_csv(self.ingestion_config.raw_data_path)
            self.logger.log(f'Saved raw data under {self.ingestion_config.raw_data_path}', 'INFO')

            # split the data into train and test set
            self.logger.log('Splitting the data into train and test set...', 'INFO')
            train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
            self.logger.log('Completed splitting the data into train and test set', 'INFO')

            # make directory for train data
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            # write train data as csv
            train_data.to_csv(self.ingestion_config.train_data_path, header=True, index=False)
            self.logger.log(f'Saved train data under {self.ingestion_config.train_data_path}', 'INFO')

            # make directory for test data
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok=True)
            # write train data as csv
            test_data.to_csv(self.ingestion_config.test_data_path, header=True, index=False)
            self.logger.log(f'Saved test data under {self.ingestion_config.test_data_path}', 'INFO')

            return (self.ingestion_config.train_data_path, 
                    self.ingestion_config.test_data_path)

        except Exception as e:
            self.logger.log('Error occurred during data ingestion', 'ERROR')      
            raise e
