import argparse
import os
from ml_project.main import main
from ml_project.config import Files

FILES = Files(
    model_path='./outputs/gradient_boosting.pkl',
    data_path='./ml_project/data/heart_cleveland_upload.csv',
    stats_path='./outputs/column_mean_std_stats.sav',
    results_path='./outputs/prediction.csv',
    log_path='./outputs/file.log'
    )
SPLIT_PARAMS = {'shuffle': True, 'test_size': 0.2, 'random_state': 42}

GRAD_BOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': 42
    }
LOG_REG_PARAMS = {'C': 10, 'penalty': 'l2', 'max_iter': 100, 'random_state': 42}


TRAINING_NAMESPACE_GRAD_BOOST = argparse.Namespace(
    model='GradientBoosting',
    params=GRAD_BOOST_PARAMS,
    train_test_split=SPLIT_PARAMS,
    inference=False,
    debug=True,
    files=FILES
)


def test_train_grad_boosting(capsys):
    main(TRAINING_NAMESPACE_GRAD_BOOST)
    captured = capsys.readouterr()
    assert '' == captured.err
    assert 'accuracy' in captured.out
    assert 'roc_auc' in captured.out
    assert 'f1_score' in captured.out
    assert os.path.exists('./outputs/gradient_boosting.pkl')


INFERENCE_NAMESPACE_GRAD_BOOST = argparse.Namespace(
    model='GradientBoosting',
    params=GRAD_BOOST_PARAMS,
    train_test_split=SPLIT_PARAMS,
    inference=True,
    debug=True,
    files=FILES
)


def test_inference_grad_boosting(capsys):
    main(INFERENCE_NAMESPACE_GRAD_BOOST)
    captured = capsys.readouterr()
    assert '' == captured.err
    assert os.path.exists('./outputs/prediction.csv')


TRAINING_NAMESPACE_LOG_REG = argparse.Namespace(
    model='LogisticRegression',
    params=LOG_REG_PARAMS,
    train_test_split=SPLIT_PARAMS,
    inference=False,
    debug=True,
    files=FILES
)


def test_train_log_reg(capsys):
    main(TRAINING_NAMESPACE_LOG_REG)
    captured = capsys.readouterr()
    assert 'accuracy' in captured.out
    assert 'roc_auc' in captured.out
    assert 'f1_score' in captured.out
    assert os.path.exists('./outputs/logistic_regression.pkl')


INFERENCE_NAMESPACE_LOG_REG = argparse.Namespace(
    model='LogisticRegression',
    params=LOG_REG_PARAMS,
    train_test_split=SPLIT_PARAMS,
    inference=False,
    debug=True,
    files=FILES
)


def test_inference_log_reg(capsys):
    main(INFERENCE_NAMESPACE_LOG_REG)
    captured = capsys.readouterr()
    assert '' == captured.err
    assert os.path.exists('./outputs/prediction.csv')
