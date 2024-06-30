from src.logger import Logger
from src.utils import save_obj
from src.utils import evaluate_model
import os, sys
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from dataclasses import dataclass

@dataclass
class ModelTrainerConfg:

    """
    Configuration for model training 

    Attributes:
        trained_model_file_path: str
            Path to the trained model.
     
    """
    trained_model_file_path = os.path.join('../artifacts', 'model.pkl')

class ModelTrainer:
    """
    Class to handle model training, evaluation and saving the best model.

    Methods:
    --------
    __init__: 
        Initializes ModelTrainer with configuration and logger.

    initiate_model_training(X_train, Y_train, X_test, Y_test): 
        Trains multiple models and evaluates them using AUC-ROC score and saves the best model. 
    """

    def __init__(self):
        """
        Initializes ModelTrainer with configuration and logger.
        """

        self.model_trainer_config = ModelTrainerConfg()
        self.logger = Logger()

    def initiate_model_training(self, X_train, Y_train, X_test, Y_test):
        """
        Trains multiple models and evaluates them using AUC-ROC score and saves the best model. 

        Args:
        X_train: pd.DataFrame
            Training features.
        Y_train: pd.Series
            Training target.
        X_test: pd.DataFrame
            Test features.
        Y_test: pd.Series
            Test target.

        Raises:
        Exception
            If any error occurs during model training or saving.     
        """
        try:
            self.logger.log('Initiating model training...')

            models = {
                'SVC': SVC(probability=True),
                'Logistic regression': LogisticRegression(),
                'Random Forest': RandomForestClassifier(),
                'XGBoost': XGBClassifier(use_label_encoder=False),
                'AdaBoost': AdaBoostClassifier(),
                'GradientBoost': GradientBoostingClassifier(),
                
            }

            model_score_dict = evaluate_model(X_train, Y_train, X_test, Y_test, models)

            print('Model name: Score')
            for key, value in model_score_dict.items():
                print(key, ':', value)
            print('\n')   

            # Find the best model based on AUC-ROC score
            best_model_score = max(sorted(model_score_dict.values())) 
            best_model_name = best_model_name = list(model_score_dict.keys())[list(model_score_dict.values()).index(best_model_score)]

            best_model = models[best_model_name]

            print(f'Best model name: {best_model_name} with AUC-ROC score: {best_model_score}\n')

            self.logger.log(f'Best model name: {best_model_name} with AUC-ROC score: {best_model_score}')

            self.logger.log('Saving the best model...')

            save_obj(self.model_trainer_config.trained_model_file_path, best_model)


        except Exception as e:
            self.logger.log('Error occurred during model training', 'ERROR')
            raise e

