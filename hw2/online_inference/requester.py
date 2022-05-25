import click

import pandas as pd
import requests

from online_inference.feature_extraction.feature_extraction import (
    load_stats
)


def load_data(path: str) -> pd.DataFrame:
    """
    Load dataset
    :param path: filepath to *.csv dataset
    :return: pandas dataframe
    """
    data = pd.read_csv(path)
    return data

@click.command()
@click.option("--data_path", default='./data/heart_cleveland_upload.csv')
@click.option("--batch_size", default=10)
@click.option("--host", default='http://127.0.0.1:8000/predict')
def requester(data_path: str, batch_size: int, host: str) -> None:
    data = load_data(data_path)
    for i in range(0, len(data), batch_size):
        instance = data.iloc[i:i + batch_size].to_dict(orient='list')
        response = requests.get(
            host,
            json={'data': instance}
        )
        print("Data_send: {} Send status: {} Output: {}".format(
            instance,
            response.status_code,
            response.content
            )
            )
        break





if __name__ == '__main__':
    requester()