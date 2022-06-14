import os

import click
import pandas as pd
from sklearn.datasets import make_classification

FEATURES_PATH = "features.csv"
TARGETS_PATH = "target.csv"

@click.command()
@click.option("--out_dir")
def main(out_dir: str) -> None:
    features, targets = make_classification(random_state=42)
    os.makedirs(out_dir, exist_ok=True)
    features = pd.DataFrame(features)
    targets = pd.DataFrame(targets, columns=["target"])
    features.to_csv(os.path.join(out_dir, FEATURES_PATH), index=False)
    targets.to_csv(os.path.join(out_dir, TARGETS_PATH), index=False)

if __name__ == '__main__':
    main()