import logging

import hydra
from hydra.core.config_store import ConfigStore

from ml_project import utils
from ml_project.feature_extraction import feature_extraction
from ml_project.model import models
from ml_project.metrics import metrics
from ml_project.config import ModelCfg
from ml_project.utils import setup_logger, LOGGER

cs = ConfigStore.instance()
cs.store(name="model_cfg", node=ModelCfg)


@hydra.main(config_path="configs", config_name='grad_boosting')
def main(cfg: ModelCfg) -> None:
    # setup logger
    level = logging.DEBUG if cfg.debug else logging.INFO
    setup_logger(
        out_file=cfg.files.log_path,
        stdout_level=level,
        file_level=level
        )
    # loading data and creating model
    data = utils.load_data(cfg.files.data_path)
    model = models.ClassificationModel(
        cfg.model,
        **cfg.params
        )
    # inference or training
    if cfg.inference:
        model.load_weights(cfg.files.model_path)
        features = feature_extraction.feature_extraction(data, cfg.model, cfg.files.stats_path, train=False)
        y_pred = model.predict(features['features'])
        utils.save_predictions(y_pred, cfg.files.results_path)
        LOGGER.info(f'Saved predictions to {cfg.files.results_path}')
    else:
        features_train, features_test = feature_extraction.feature_extraction_with_test_train_split(
            data,
            cfg.model,
            cfg.files.stats_path,
            **cfg.train_test_split
        )
        model.fit(features_train['features'], features_train['labels'])
        model.save_weights(cfg.files.model_path)
        y_pred = model.predict(features_test['features'])
        model_metrics = metrics.get_metrics(features_test['labels'], y_pred)
        LOGGER.info(f'Metrics evaluation on training data {model_metrics}')


if __name__ == '__main__':
    main()
