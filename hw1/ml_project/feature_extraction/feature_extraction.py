from typing import Dict, List, Tuple, Union
import pickle


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from ml_project.utils import LOGGER

TARGET_COLUMN = 'condition'
LOG_SCALE_COLUMN = 'oldpeak'


def one_hot_encoding_by_column(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Encode categorical data in column using pd.get_dummies
    :param data: input data
    :param column: input column
    :return: dataframe with one_hot_encoded column
    """
    one_hot = pd.get_dummies(data[column], prefix=column)
    data = data.drop(column, axis=1)
    data = data.join(one_hot)
    return data


def divide_columns_by_type(data: pd.DataFrame, threshold: int = 10) -> Dict[str, List[str]]:
    """
    Divide columns into categorical and continuous
    by threshold of unique values in column
    :param data: input data
    :param threshold: threshold of unique values in column
    :return: {'categorical': list_of_columns, 'continuous': list_of_columns,}
    """
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


def save_stats(stats: dict, filename: str) -> None:
    """
    Save stats to pkl file
    :param stats: dict of stats by column key
    :param filename: filepath to save
    """
    LOGGER.info(f'Saving split into categorical and continuous groups and statistics: means and stds to {filename}')
    with open(filename, 'wb') as file:
        pickle.dump(stats, file)


def load_stats(filename: str) -> dict:
    """
    Load stats from pkl file
    :param filename: filepath to load
    :return: dictionary of stats by column key
    """
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
    """
    Extraction features and labels (if exists) for training or inference
    :param data: input data
    :param model_type: log_reg or gradboosting, for gradboosting no preprocessing
    :param filename: where to save stats of feature_extraction
    :param train: train or inference
    :return: {'features': features data, 'labels': labels data}
    """
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


def feature_extraction_with_test_train_split(
    data: pd.DataFrame,
    model_type: str,
    filename: str,
    **kwards
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Feature exptraction with splitting data to train and validation
    :param data: input data
    :model_type: type of model, need for feature engineering
    :filename: filepath to save stats
    :kwards: model parameters
    """
    data_train, data_valid = train_test_split(data, **kwards)
    features_train = feature_extraction(data_train, model_type, filename, train=True)
    features_valid = feature_extraction(data_valid, model_type, filename, train=False)
    return features_train, features_valid
