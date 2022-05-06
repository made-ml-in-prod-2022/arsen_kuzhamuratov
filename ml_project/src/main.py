import os
import argparse

import utils


def main(args):
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
        # replace with logger
        print(utils.get_metrics(features['labels'], y_pred))


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
    args = parser.parse_args()
    main(args)
