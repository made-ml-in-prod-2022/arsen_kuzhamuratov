import argparse
import os
import logging

from utils import setup_logger, LOGGER
import utils


def main(args):
    level = logging.DEBUG if args.debug else logging.INFO
    setup_logger(
        out_file='../logging_data/file.log',
        stdout_level=level,
        file_level=level
        )
    data = utils.load_data(args.data_path)
    features = utils.feature_extraction(data, args.stats_path, args.train)
    model = utils.LogisticRegressionModel(args.C, args.model_path)
    if args.inference:
        model.load_weights()
        y_pred = model.predict(features['features'])
        utils.save_predictions(y_pred, args.results_path)

    else:
        model.fit(features['features'], features['labels'])
        y_pred = model.predict(features['features'])
        metrics = utils.get_metrics(features['labels'], y_pred)
        LOGGER.info(f'Metrics evaluation on training data {metrics}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode_type = parser.add_mutually_exclusive_group()
    mode_type.add_argument('--train', action='store_true', help='Train model')
    mode_type.add_argument('--inference', action='store_true', help='Inference model')
    parser.add_argument('-d', '--data_path', default='../data/heart_cleveland_upload.csv', help='Path to dataset')
    parser.add_argument('-C', default=10, help='l2 regularization parameter')
    # add only for inference
    parser.add_argument(
        '--stats_path',
        default='../eda_stats/column_mean_std_stats.sav',
        help='Path statistical train data'
        )
    parser.add_argument(
        '--model_path',
        default='../models/model.pkl',
        help='Path statistical train data'
        )
    parser.add_argument(
        '--results_path',
        default='../results/results.csv',
        help='Path to save predictions'
        )
    parser.add_argument(
        '--debug', action='store_true'
    )
    args = parser.parse_args()
    main(args)
