import pandas as pd
from feature_extraction import feature_extraction
from typing import Dict, List, Union
SCALAR = Union[float, int]
JSON_TYPE = Dict[str, Union[SCALAR, List[SCALAR]]]
REAL_COLUMNS = ['age', 'sex', 'cp', 'trestbps',
                'chol', 'fbs', 'restecg', 'thalach',
                'exang', 'oldpeak', 'slope', 'ca', 'thal', 'condition']

X = {column: [0] for column in REAL_COLUMNS}
X = pd.DataFrame.from_dict(X)
model_type = 'GradientBoosting'
filepath = './artifacts/artifacts/column_mean_std_stats.sav'

print(feature_extraction.feature_extraction(X, model_type, filepath))
