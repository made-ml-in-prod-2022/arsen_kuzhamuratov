import pickle
from typing import Union


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


class ClassificationModel:
    """
    Main class for fitting, predicting, saving and
    loading weights of sklearn logistic regression
    """
    def __init__(self, filepath) -> None:
        """
        Initializing model with params
        :param filepath: path to trained model
        """
        self.filepath = filepath
        self.load_weights(self.filepath)

    def predict(self, X: Union[pd.DataFrame, np.array]) -> Union[pd.DataFrame, np.array]:
        """
        Predict targets on features
        :param X: input features
        :return: predictions
        """
        return self.clf.predict(X)

    def load_weights(self, filename: str) -> None:
        """
        Load model
        :param filename: filepath to load *.pkl
        """
        with open(filename, 'rb') as file:
            self.clf = pickle.load(file)
