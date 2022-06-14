import os
import pickle

import click
import pandas as pd

INFERENCE_DATAPATH = "features.csv"

@click.command()
@click.option("--in_dir")
@click.option("--model_path")
@click.option("--pred_path")
def main(in_dir: str, model_path: str, pred_path: str) -> None:
    path = os.path.join(in_dir, INFERENCE_DATAPATH)
    data = pd.read_csv(path)
    
    with open(model_path, 'rb') as fin:
        model = pickle.load(fin)

    predictions = model.predict(data)

    pred = pd.DataFrame(predictions)
    pred.to_csv(pred_path, index=False)

if __name__ == '__main__':
    main()