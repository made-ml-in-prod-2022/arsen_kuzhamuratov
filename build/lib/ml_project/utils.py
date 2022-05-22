import logging
import sys
from typing import Dict, List, Optional, Tuple, Union


import pickle
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


LOGGER = logging.getLogger()
FORMATTER = logging.Formatter("%(asctime)s - %(levelname)s - %(funcName)s - %(message)s")

TARGET_COLUMN = 'condition'
LOG_SCALE_COLUMN = 'oldpeak'

def setup_logger(
    out_file: Optional[str] =None,
     stdout: bool=True,
      stdout_level: int=logging.INFO,
       file_level: int=logging.DEBUG
       ) -> logging.RootLogger:
    LOGGER.handlers = []
    LOGGER.setLevel(min(stdout_level, file_level))

    if stdout:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(FORMATTER)
        handler.setLevel(stdout_level)
        LOGGER.addHandler(handler)

    if out_file is not None:
        handler = logging.FileHandler(out_file)
        handler.setFormatter(FORMATTER)
        handler.setLevel(file_level)
        LOGGER.addHandler(handler)

    LOGGER.info("logger set up")
    return LOGGER


class ClassificationModel:
    """
    Main class for fitting, predicting, saving and
    loading weights of sklearn logistic regression
    """
    def __init__(self, model_type, **kwards):
        # add params
        LOGGER.info(f'Creating {model_type} with {kwards}')
        self.clf = (LogisticRegression(**kwards) if model_type == 'LogisticRegression'
                    else GradientBoostingClassifier(**kwards))

    def fit(self, X: Union[pd.DataFrame, np.array], y: Union[pd.DataFrame, list, np.array]) -> None:
        # check x, y
        LOGGER.info(
            f'Fitting model to dataset with feature sizes: {len(X)} x {len(X.columns)} and labels size: {len(y)}'
            )
        self.clf.fit(X, y)

    def predict(self, X: Union[pd.DataFrame, np.array]) -> Union[pd.DataFrame, np.array]:
        # check input size
        LOGGER.info(f'Predicting model on dataset with sizes: {len(X)} x {len(X.columns)}')
        return self.clf.predict(X)

    def save_weights(self, filename: str) -> None:
        LOGGER.debug(f'Saving model weights to {filename}')
        with open(filename, 'wb') as file:
            pickle.dump(self.clf, file)

    def load_weights(self, filename: str) -> None:
        LOGGER.debug(f'Downloading model weights to {filename}')
        with open(filename, 'rb') as file:
            self.clf = pickle.load(file)


def one_hot_encoding_by_column(data: pd.DataFrame, column: str) -> pd.DataFrame:
    one_hot = pd.get_dummies(data[column], prefix=column)
    data = data.drop(column, axis=1)
    data = data.join(one_hot)
    return data


def divide_columns_by_type(data: pd.DataFrame, threshold: int = 10) -> Dict[str, List[str]]:
    LOGGER.debug(f'Dividing columns into categorical and continuous groups by {threshold}')
    categorical_val = []
    continuous_val = []
    for column in data.columns:
        if len(data[column].unique()) <= threshold:
            categorical_val.append(column)
        else:
            continuous_val.append(column)
    LOGGER.debug(f'Categorical columns: {categorical_val}; Continuous columns: {continuous_val}')
    return {
        'categorical': categorical_val,
        'continuous': continuous_val
        }


def load_data(path: str) -> pd.DataFrame:
    LOGGER.info(f'Downloading data from {path}')
    data = pd.read_csv(path)
    LOGGER.debug(f'Table data has length: {len(data)} and number of columns: {len(data.columns)}')
    return data


def save_predictions(data: pd.DataFrame, filename: str) -> None:
    LOGGER.info(f'Saving predictions with size {len(data)} for inference to {filename}')
    data = pd.DataFrame(data, columns=['prediction'])
    data.to_csv(filename, index=False)


def save_stats(stats: dict, filename: str) -> None:
    LOGGER.info(f'Saving split into categorical and continuous groups and statistics: means and stds to {filename}')
    with open(filename, 'wb') as file:
        pickle.dump(stats, file)


def load_stats(filename: str) -> dict:
    LOGGER.info(f'Loading split into categorical and continuous groups and statistics from {filename}')
    with open(filename, 'rb') as file:
        stats = pickle.load(file)
    return stats


def feature_extraction(
    data: pd.DataFrame, 
    model_type: str, 
    filename: str, 
    train: bool = True
    ) -> Dict[str, Union[pd.DataFrame, np.array]]:
    if TARGET_COLUMN in data.columns:
        y_true = data[TARGET_COLUMN]
        X = data.drop(columns=[TARGET_COLUMN])
    else:
        y_true = None
        X = data
    X[LOG_SCALE_COLUMN] = X[LOG_SCALE_COLUMN].apply(lambda x: np.log(1 + x))

    if model_type == 'GradientBoosting':
        LOGGER.debug("Prepare dataset for training GradientBoostingClassifier...")
        return {
            'labels': y_true,
            'features': X
        }

    LOGGER.debug("Prepare dataset for LogisticRegression...")
    if train:
        column_type_dict = divide_columns_by_type(X)
        for column in column_type_dict['categorical']:
            X = one_hot_encoding_by_column(X, column)

        LOGGER.debug(f'Data shape after one hot encoding: {len(X.columns)}')
        mean_columns = {}
        std_columns = {}
        for column in column_type_dict['continuous']:
            mean_columns[column] = np.std(X[column])
            std_columns[column] = np.mean(X[column])

        eda_stats = {
            'column_type': column_type_dict,
            'columns': X.columns,
            'means': mean_columns,
            'stds': std_columns
        }
        save_stats(eda_stats, filename)
    else:
        LOGGER.debug("Prepare dataset for inference...")

        eda_stats = load_stats(filename)
        column_type_dict = eda_stats['column_type']
        for column in column_type_dict['categorical']:
            X = one_hot_encoding_by_column(X, column)
        X = X.reindex(columns=eda_stats['columns'], fill_value=0)

        LOGGER.debug(f'Data shape after one hot encoding: {len(X.columns)}')
    for column in column_type_dict['continuous']:
        X[column] = X[column].apply(lambda x: (x - eda_stats['means'][column]) / (eda_stats['stds'][column] + 1e-8))

    return {
        'features': X,
        'labels': y_true
        }


def feature_extraction_with_test_train_split(data: pd.DataFrame, model_type: str, filename: str, **kwards) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_train, data_valid = train_test_split(data, **kwards)
    features_train = feature_extraction(data_train, model_type, filename, train=True)
    features_valid = feature_extraction(data_valid, model_type, filename, train=False)
    return features_train, features_valid


def get_metrics(
    y_true: Union[pd.DataFrame, list, np.array], 
    y_pred: Union[pd.DataFrame, list, np.array]
    ) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred)
    }
