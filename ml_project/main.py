import argparse
import logging

import yaml

from ml_project import utils
from ml_project.utils import setup_logger, LOGGER


def main(args):
    # loading model config
    with open(args.config_path) as file:
        model_config = yaml.load(file, Loader=yaml.FullLoader)

    # setup logger
    level = logging.DEBUG if args.debug else logging.INFO
    setup_logger(
        out_file=args.log_path,
        stdout_level=level,
        file_level=level
        )
    # loading data and creating model
    data = utils.load_data(args.data_path)
    model = utils.ClassificationModel(
        model_config['model'],
        model_config['model_path'], **model_config['params']
        )
    # inference or training
    if args.inference:
        model.load_weights()
        features = utils.feature_extraction(data, model_config['model'], args.stats_path, args.train)
        y_pred = model.predict(features['features'])
        utils.save_predictions(y_pred, args.results_path)
        LOGGER.info(f'Saved predictions to {args.results_path}')
    else:
        features_train, features_test = utils.feature_extraction_with_test_train_split(
            data,
            model_config['model'],
            args.stats_path,
            **model_config['train_test_split']
        )
        model.fit(features_train['features'], features_train['labels'])
        y_pred = model.predict(features_test['features'])
        metrics = utils.get_metrics(features_test['labels'], y_pred)
        LOGGER.info(f'Metrics evaluation on training data {metrics}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode_type = parser.add_mutually_exclusive_group()
    mode_type.add_argument('--train', action='store_true', help='Train model')
    mode_type.add_argument('--inference', action='store_true', help='Inference model')
    parser.add_argument('config_path')
    parser.add_argument('-d', '--data_path', default='./ml_project/data/heart_cleveland_upload.csv', help='Path to dataset')

    parser.add_argument(
        '--stats_path',
        default='./outputs/column_mean_std_stats.sav',
        help='Path statistical train data'
        )

    parser.add_argument(
        '--results_path',
        default='./outputs/prediction.csv',
        help='Path to save predictions'
        )
    parser.add_argument(
        '--log_path',
        default='./outputs/file.log',
        help='Path to save logs'
        )
    parser.add_argument(
        '--debug', action='store_true'
    )
    args = parser.parse_args()
    main(args)
