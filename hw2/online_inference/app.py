import os
from typing import Dict


import yaml
import uvicorn
from fastapi import FastAPI


from online_inference.config import ModelCfg, RequestData
from online_inference.model.models import ClassificationModel
from online_inference.feature_extraction.feature_extraction import load_stats
from online_inference.model.predict import get_predict

CONFIG_PATH = ''
DEFAULT_CONFIG_PATH = './configs/log_reg.yaml'
app = FastAPI()
artifacts = {}


@app.get("/")
async def root() -> Dict[str, str]:
    return {"message": "Root position"}


@app.on_event('startup')
async def startup_event() -> None:
    """
    Prepare artifacts: loading model and eda_stats
    """
    model_config = os.getenv(CONFIG_PATH) or DEFAULT_CONFIG_PATH
    config = ModelCfg(
        **yaml.safe_load(open(model_config))
    )
    artifacts['model_name'] = config.model_type
    artifacts['model_state'] = ClassificationModel(config.pathes['model_path'])
    if config.pathes['stats_path']:
        artifacts['feature_extraction_stats'] = load_stats(config.pathes['stats_path'])
    else:
        artifacts['feature_extraction_stats'] = None


@app.get('/predict', response_model=Dict[str, int])
async def predict(request: RequestData):
    """
    Predict with model on requests
    :param request: request with data field in json format
    """
    return get_predict(artifacts, request.data)


@app.get('/health')
async def check_ready() -> bool:
    """
    Check readiness of server: artifacts dowloaded
    :return: status
    """
    return len(artifacts) == 3


def run() -> None:
    uvicorn.run("app:app", host='0.0.0.0', port=8000)


if __name__ == "__main__":
    run()
