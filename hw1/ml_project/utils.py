import logging
import sys
from typing import Optional


import pandas as pd


LOGGER = logging.getLogger()
FORMATTER = logging.Formatter("%(asctime)s - %(levelname)s - %(funcName)s - %(message)s")


def setup_logger(
    out_file: Optional[str] = None,
    stdout: bool = True,
    stdout_level: int = logging.INFO,
    file_level: int = logging.DEBUG
    ) -> logging.RootLogger:
    """
    Setup logger
    :param out_file: filepath to *.log file
    :param stdout: print to console or not
    :param stdout_level: choose logging level to stdout logs
    :param file_level: choose logging level to file logs
    :return: logger object
    """
    LOGGER.handlers = []
    LOGGER.setLevel(min(stdout_level, file_level))

    if stdout:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(FORMATTER)
        handler.setLevel(stdout_level)
        LOGGER.addHandler(handler)

    if out_file is not None:
        handler = logging.FileHandler(out_file)
        handler.setFormatter(FORMATTER)
        handler.setLevel(file_level)
        LOGGER.addHandler(handler)

    LOGGER.info("logger set up")
    return LOGGER


def load_data(path: str) -> pd.DataFrame:
    """
    Load dataset
    :param path: filepath to *.csv dataset
    :return: pandas dataframe
    """
    LOGGER.info(f'Downloading data from {path}')
    data = pd.read_csv(path)
    LOGGER.debug(f'Table data has length: {len(data)} and number of columns: {len(data.columns)}')
    return data


def save_predictions(data: pd.DataFrame, filename: str) -> None:
    """
    Save predictions to csv file
    :param data: predictions pd.DataFrame object
    :param filename: filepath to save
    """
    LOGGER.info(f'Saving predictions with size {len(data)} for inference to {filename}')
    data = pd.DataFrame(data, columns=['prediction'])
    data.to_csv(filename, index=False)
