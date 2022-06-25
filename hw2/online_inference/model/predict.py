from typing import Dict, List, Union
from online_inference.feature_extraction.feature_extraction import feature_extraction


SCALAR = Union[float, int]
JSON_TYPE = Dict[str, Union[SCALAR, List[SCALAR]]]


def get_predict(artifacts, input_data: JSON_TYPE) -> Dict[str, int]:
    features = feature_extraction(
        input_data,
        artifacts['model_name'],
        artifacts['feature_extraction_stats']
        )
    return {
        'Id' + str(id): label for id, label in enumerate(
            artifacts['model_state'].predict(features)
            )
            }
