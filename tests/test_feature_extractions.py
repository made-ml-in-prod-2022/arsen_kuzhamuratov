import pandas as pd
import numpy as np


from ml_project.feature_extraction import feature_extraction

FILENAME = './file.sav'

REAL_COLUMNS = ['age', 'sex', 'cp', 'trestbps',
                'chol', 'fbs', 'restecg', 'thalach',
                'exang', 'oldpeak', 'slope', 'ca', 'thal', 'condition']
FAKE_DATA_SMALL = pd.DataFrame.from_dict(
    {
        'a1': [1] * 10 + [2] * 10,
        'a2': [1] * 20,
        'a3': list(range(20)),
        'a4': list(range(9)) + [0] * 11
        }
    )

FAKE_DATA_MID = pd.DataFrame.from_dict({
    column: list(range(ind)) + [0] * (len(REAL_COLUMNS) - ind)
    for ind, column in enumerate(REAL_COLUMNS, 1)
    })
FAKE_DATA_MID['condition'] = np.random.randint(2, size=len(REAL_COLUMNS))

SINGLE_LINE = [0] * (len(REAL_COLUMNS) - 1)
VALID_SINGLE_LINE = pd.DataFrame.from_dict({
    column: [value]
    for column, value in zip(REAL_COLUMNS[:-1], SINGLE_LINE)
    })


def test_divide_columns_by_type():
    result = feature_extraction.divide_columns_by_type(FAKE_DATA_SMALL, threshold=10)
    assert ['a1', 'a2', 'a4'] == result['categorical']
    assert ['a3'] == result['continuous']


def test_divide_columns_by_type_mid_dataset():
    result = feature_extraction.divide_columns_by_type(FAKE_DATA_MID, threshold=10)
    assert 11 == len(result['categorical'])
    assert 3 == len(result['continuous'])


def test_train_feature_extraction_grad_boosting():
    features = feature_extraction.feature_extraction(
        data=FAKE_DATA_MID,
        model_type='GradientBoosting',
        filename=FILENAME,
        train=True)
    assert 14 == len(features['labels'])
    assert np.where(FAKE_DATA_MID.drop(columns=['condition']) == features['features'])


def test_train_feature_extraction_log_reg():
    features = feature_extraction.feature_extraction(
        data=FAKE_DATA_MID,
        model_type='LogisticRegression',
        filename=FILENAME,
        train=True)
    assert 14 == len(features['labels'])
    assert 55 + 3 == len(features['features'].columns)


def test_valid_feature_extraction_log_reg():
    features = feature_extraction.feature_extraction(
        data=FAKE_DATA_MID,
        model_type='LogisticRegression',
        filename=FILENAME,
        train=False)
    assert 14 == len(features['labels'])
    assert 55 + 3 == len(features['features'].columns)


def test_valid_feature_extraction_log_reg_single_line_input():
    features = feature_extraction.feature_extraction(
        data=VALID_SINGLE_LINE,
        model_type='LogisticRegression',
        filename=FILENAME,
        train=False)
    assert features['labels'] is None
    assert 55 + 3 == len(features['features'].columns)
    assert 5 == np.unique(features['features'].values).shape[0]
