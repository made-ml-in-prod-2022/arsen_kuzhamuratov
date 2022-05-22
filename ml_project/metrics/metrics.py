from typing import Dict, Union

import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score
    )


def get_metrics(
    y_true: Union[pd.DataFrame, list, np.array],
    y_pred: Union[pd.DataFrame, list, np.array]
    ) -> Dict[str, float]:
    """
    Calculating metrics using sklearn
    :param y_true: true labels
    :param y_pred: predictions
    :return: dict of metrics {'metric': value, etc.}
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred)
    }
