import pandas as pd
import numpy as np

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import ml_project

PATH_TO_REAL_DATASET = '../ml_project/data/heart_cleveland_upload.csv'
REAL_COLUMNS = pd.read_csv(PATH_TO_REAL_DATASET).columns
FAKE_DATA_SMALL = pd.DataFrame.from_dict(
    {column: [0] for column in REAL_COLUMNS}
    )

FAKE_DATA_BIG = pd.DataFrame.from_dict({
    column: np.random.randint(low=0, high=1 + np.random.randint(20), size=1000) 
    for column in REAL_COLUMNS
    })


def test_logistic_regression_model():
    pass

def test_one_hot_encoding():
    pass

def test_divide_columns_by_type():
    pass

def test_feature_extraction():
    pass

def test_feature_extraction_big():
    pass
