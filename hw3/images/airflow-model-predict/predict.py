import os
import pickle

import click
import pandas as pd


INFERENCE_DATAPATH = "features.csv"
MODEL_PATH = "model.pkl"
PREDS_PATH = "predictions.csv"


@click.command()
@click.option("--in_dir")
@click.option("--model_dir")
@click.option("--pred_dir")
def main(in_dir: str, model_dir: str, pred_dir: str) -> None:
    path = os.path.join(in_dir, INFERENCE_DATAPATH)
    data = pd.read_csv(path)
    
    model_path = os.path.join(model_dir, MODEL_PATH)
    with open(model_path, 'rb') as fin:
        model = pickle.load(fin)
    predictions = model.predict(data)

    os.makedirs(pred_dir, exist_ok=True)
    pred = pd.DataFrame(predictions)
    pred_path = os.path.join(pred_dir, PREDS_PATH)
    pred.to_csv(pred_path, index=False)


if __name__ == '__main__':
    main()
