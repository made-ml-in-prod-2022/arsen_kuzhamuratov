import argparse
import os
from ml_project.main import main


TRAINING_NAMESPACE_GRAD_BOOST = argparse.Namespace(
    config_path='./configs/gradient_boosting.yaml',
    data_path='./ml_project/data/heart_cleveland_upload.csv',
    train=True,
    debug=True,
    inference=False,
    stats_path='./outputs/column_mean_std_stats.sav',
    results_path='./outputs/prediction.csv',
    log_path='./outputs/file.log',
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
    config_path='./configs/gradient_boosting.yaml',
    data_path='./ml_project/data/heart_cleveland_upload.csv',
    train=False,
    debug=True,
    inference=True,
    stats_path='./outputs/column_mean_std_stats.sav',
    results_path='./outputs/prediction.csv',
    log_path='./outputs/file.log',
    )


def test_inference_grad_boosting(capsys):
    main(INFERENCE_NAMESPACE_GRAD_BOOST)
    captured = capsys.readouterr()
    assert '' == captured.err
    assert os.path.exists('./outputs/prediction.csv')


TRAINING_NAMESPACE_LOG_REG = argparse.Namespace(
    config_path='./configs/logistic_regression.yaml',
    data_path='./ml_project/data/heart_cleveland_upload.csv',
    train=True,
    debug=True,
    inference=False,
    stats_path='./outputs/column_mean_std_stats.sav',
    results_path='./outputs/prediction.csv',
    log_path='./outputs/file.log',
    )


def test_train_log_reg(capsys):
    main(TRAINING_NAMESPACE_LOG_REG)
    captured = capsys.readouterr()
    assert 'accuracy' in captured.out
    assert 'roc_auc' in captured.out
    assert 'f1_score' in captured.out
    assert os.path.exists('./outputs/logistic_regression.pkl')

INFERENCE_NAMESPACE_LOG_REG = argparse.Namespace(
    config_path='./configs/logistic_regression.yaml',
    data_path='./ml_project/data/heart_cleveland_upload.csv',
    train=False,
    debug=True,
    inference=True,
    stats_path='./outputs/column_mean_std_stats.sav',
    results_path='./outputs/prediction.csv',
    log_path='./outputs/file.log',
    )


def test_inference_log_reg(capsys):
    main(INFERENCE_NAMESPACE_LOG_REG)
    captured = capsys.readouterr()
    assert '' == captured.err
    assert os.path.exists('./outputs/prediction.csv')
