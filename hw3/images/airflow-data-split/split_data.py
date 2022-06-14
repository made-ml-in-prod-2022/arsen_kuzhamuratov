import os

import click
import pandas as pd
from sklearn.model_selection import train_test_split


FEATURES_PATH = "features.csv"
TARGETS_PATH = "target.csv"

FEATURES_PATH_TRAIN = "features_train.csv"
FEATURES_PATH_VAL = "features_val.csv"

TARGETS_PATH_TRAIN = "target_train.csv"
TARGETS_PATH_VAL = "target_val.csv"

VAL_PART = 0.2

@click.command()
@click.option("--in_dir")
@click.option("--out_dir")
def main(in_dir: str, out_dir: str) -> None:
    data = pd.read_csv(os.path.join(in_dir, FEATURES_PATH))
    target = pd.read_csv(os.path.join(in_dir, TARGETS_PATH))
    X_train, X_val, y_train, y_val = train_test_split(data, target, test_size=VAL_PART, random_state=1337)
    os.makedirs(out_dir, exist_ok=True)
    X_train.to_csv(os.path.join(out_dir, FEATURES_PATH_TRAIN), index=False)
    y_train.to_csv(os.path.join(out_dir, TARGETS_PATH_TRAIN), index=False)
    X_val.to_csv(os.path.join(out_dir, FEATURES_PATH_VAL), index=False)
    y_val.to_csv(os.path.join(out_dir, TARGETS_PATH_VAL), index=False)
    

if __name__ == '__main__':
    main()