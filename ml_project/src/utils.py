import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


class LogisticRegressionModel:
    """
    Main class for fitting, predicting, saving and loading weights of
    sklearn logistic regression
    """
    def __init__(self, l2_reg_parameter, filename):
        # add params
        self.clf = LogisticRegression(C=l2_reg_parameter)
        self.filename = filename

    def fit(self, X, y):
        # check x, y
        self.clf.fit(X, y)
        self.save_weights()

    def predict(self, X):
        # check input size
        return self.clf.predict(X)

    def save_weights(self):
        with open(self.filename, 'wb') as file:
            pickle.dump(self.clf, file)

    def load_weights(self):
        with open(self.filename, 'rb') as file:
            self.clf = pickle.load(file)


def one_hot_encoding_by_column(data, column):
    one_hot = pd.get_dummies(data[column], prefix=column)
    data = data.drop(column, axis=1)
    data = data.join(one_hot)
    return data


def divide_columns_by_type(data):
    categorical_val = []
    continuous_val = []
    for column in data.columns:
        if len(data[column].unique()) <= 10:
            categorical_val.append(column)
        else:
            continuous_val.append(column)
    return {
        'categorical': categorical_val,
        'continuous': continuous_val
        }


def load_data(path):
    data = pd.read_csv(path)
    return data


def save_predictions(data, filename):
    data = pd.DataFrame(data, columns=['prediction'])
    data.to_csv(filename, index=False)


def save_stats(stats, filename):
    with open(filename, 'wb') as file:
        pickle.dump(stats, file)


def load_stats(filename):
    with open(filename, 'rb') as file:
        stats = pickle.load(file)
    return stats


def feature_extraction(data, filename, train=True):
    # check data for nulls
    if train:

        y_true = data['condition']
        X = data.drop(columns=['condition'])
        X['oldpeak'] = X['oldpeak'].apply(lambda x: np.log(1 + x))

        column_type_dict = divide_columns_by_type(X)
        for column in column_type_dict['categorical']:
            X = one_hot_encoding_by_column(X, column)

        mean_columns = {}
        std_columns = {}
        for column in column_type_dict['continuous']:
            mean_columns[column] = np.std(X[column])
            std_columns[column] = np.mean(X[column])

        eda_stats = {
            'column_type': column_type_dict,
            'means': mean_columns,
            'stds': std_columns
        }

        save_stats(eda_stats, filename)
    else:
        X = data
        X['oldpeak'] = X['oldpeak'].apply(lambda x: np.log(1 + x))
        eda_stats = load_stats(filename)

        column_type_dict = eda_stats['column_type']
        for column in column_type_dict['categorical']:
            X = one_hot_encoding_by_column(X, column)

    for column in column_type_dict['continuous']:
        X[column] = X[column].apply(lambda x: (x - eda_stats['means'][column]) / (eda_stats['stds'][column] + 1e-8))

    return {
        'features': X,
        'labels': None if not train else y_true
        }


def get_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred)
    }
