import pickle
from typing import Union


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


from ml_project.utils import LOGGER


class ClassificationModel:
    """
    Main class for fitting, predicting, saving and
    loading weights of sklearn logistic regression
    """
    def __init__(self, model_type: str, **kwards) -> None:
        """
        Initializing model with params
        :param model_type: Sklearn model
        :param kwards: possible params for model, example: lr, n_estimators, etc.
        """
        LOGGER.info(f'Creating {model_type} with {kwards}')
        self.clf = (LogisticRegression(**kwards) if model_type == 'LogisticRegression'
                    else GradientBoostingClassifier(**kwards))

    def fit(self, X: Union[pd.DataFrame, np.array], y: Union[pd.DataFrame, list, np.array]) -> None:
        """
        Fit model to dataset (X, y)
        :param X: input features
        :param y: input targets
        """
        LOGGER.info(
            f'Fitting model to dataset with feature sizes: {len(X)} x {len(X.columns)} and labels size: {len(y)}'
            )
        self.clf.fit(X, y)

    def predict(self, X: Union[pd.DataFrame, np.array]) -> Union[pd.DataFrame, np.array]:
        """
        Predict targets on features
        :param X: input features
        :return: predictions
        """
        # check input size
        LOGGER.info(f'Predicting model on dataset with sizes: {len(X)} x {len(X.columns)}')
        return self.clf.predict(X)

    def save_weights(self, filename: str) -> None:
        """
        Save model
        :param filename: filepath to save *.pkl
        """
        LOGGER.debug(f'Saving model weights to {filename}')
        with open(filename, 'wb') as file:
            pickle.dump(self.clf, file)

    def load_weights(self, filename: str) -> None:
        """
        Load model
        :param filename: filepath to load *.pkl
        """
        LOGGER.debug(f'Downloading model weights to {filename}')
        with open(filename, 'rb') as file:
            self.clf = pickle.load(file)
