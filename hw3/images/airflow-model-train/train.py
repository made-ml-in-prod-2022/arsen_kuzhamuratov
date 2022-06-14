import os
import pickle

import click
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

FEATURES_PATH_TRAIN = "features_train.csv"
TARGETS_PATH_TRAIN = "target_train.csv"
MODEL_PATH = 'model.pkl'

@click.command()
@click.option("--in_dir")
@click.option("--out_dir")
def train(in_dir: str, out_dir: str) -> None:
    features = pd.read_csv(os.path.join(in_dir, FEATURES_PATH_TRAIN))
    targets = pd.read_csv(os.path.join(in_dir, TARGETS_PATH_TRAIN))
    clf = GradientBoostingClassifier()
    clf.fit(features, targets["target"])

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, MODEL_PATH), 'wb') as fout:
        pickle.dump(clf, fout)

if __name__ == '__main__':
    train()