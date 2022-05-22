from dataclasses import dataclass


@dataclass
class Files:
    model_path: str
    data_path: str
    stats_path: str
