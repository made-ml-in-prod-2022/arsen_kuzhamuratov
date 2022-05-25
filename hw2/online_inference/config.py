from dataclasses import dataclass
from typing import Dict
from pydantic import BaseModel


@dataclass
class Files:
    model_path: str
    stats_path: str


@ dataclass
class ModelCfg:
    model_type: str
    pathes: Files


class RequestData(BaseModel):
    data: Dict[str, list]
