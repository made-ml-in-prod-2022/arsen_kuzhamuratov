from typing import Dict, List, Union
import pickle


import pandas as pd
import numpy as np


TARGET_COLUMN = 'condition'
LOG_SCALE_COLUMN = 'oldpeak'

SCALAR = Union[float, int]
JSON_TYPE = Dict[str, Union[SCALAR, List[SCALAR]]]

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
    categorical_val = []
    continuous_val = []
    for column in data.columns:
        if len(data[column].unique()) <= threshold:
            categorical_val.append(column)
        else:
            continuous_val.append(column)
    return {
        'categorical': categorical_val,
        'continuous': continuous_val
        }


def load_stats(filename: str) -> dict:
    """
    Load stats from pkl file
    :param filename: filepath to load
    :return: dictionary of stats by column key
    """
    with open(filename, 'rb') as file:
        stats = pickle.load(file)
    return stats


def cast_json_to_dataframe(X: JSON_TYPE) -> pd.DataFrame:
    for key, value in X.items():
        if not isinstance(value, list):
            X[key] = [value]
    return pd.DataFrame.from_dict(X)

def feature_extraction(
    X: Union[pd.DataFrame, JSON_TYPE],
    model_type: str,
    eda_stats: Union[dict, None] = None
    ) -> Union[pd.DataFrame, np.array]:
    """
    Extraction features for inference
    In LogReg case loaded stats from filename
    In GradBoosting no stats needs
    :param data: input data
    :param model_type: log_reg or gradboosting, for gradboosting no preprocessing
    :param filename: filepath to load stats of feature_extraction in log_reg case
    :return: features
    """
    if isinstance(X, dict):
        X = cast_json_to_dataframe(X)

    X[LOG_SCALE_COLUMN] = X[LOG_SCALE_COLUMN].apply(lambda x: np.log(1 + x))

    if model_type == 'GradientBoosting':
        return X
    elif model_type == 'LogisticRegression':
        column_type_dict = eda_stats['column_type']
        for column in column_type_dict['categorical']:
            X = one_hot_encoding_by_column(X, column)
        X = X.reindex(columns=eda_stats['columns'], fill_value=0)

        for column in column_type_dict['continuous']:
            X[column] = X[column].apply(
                lambda x: (x - eda_stats['means'][column]) / (eda_stats['stds'][column] + 1e-8)
                )
        return X
    else:
        raise NotImplementedError(f"{model_type} not implemented...")
