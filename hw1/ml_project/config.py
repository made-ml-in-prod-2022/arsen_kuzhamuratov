from dataclasses import dataclass


@dataclass
class Files:
    model_path: str
    data_path: str
    stats_path: str
    results_path: str
    log_path: str


@dataclass
class SplittingParams:
    shuffle: bool
    test_size: float
    random_state: int


@dataclass
class ModelParams:
    C: float
    penalty: str
    max_iter: int
    n_estimators: int
    max_depth: int
    random_state: int


@dataclass
class ModelCfg:
    model: str
    params: ModelParams
    train_test_split: SplittingParams
    inference: bool
    debug: bool
    files: Files
