from fastapi.testclient import TestClient


from online_inference.app import app


REAL_COLUMNS = ['age', 'sex', 'cp', 'trestbps',
                'chol', 'fbs', 'restecg', 'thalach',
                'exang', 'oldpeak', 'slope', 'ca', 'thal', 'condition']

CHECK_INSTANCE = {column: [0] for column in REAL_COLUMNS}

BATCH_INSTANCE = {column: [0, 1, 2, 3, 4] for column in REAL_COLUMNS}

def test_startup():
    with TestClient(app) as client:
        response = client.get('/')
    assert 200 == response.status_code


artifacts = {'model_name': 1, 'model_state': 2, 'feature_extraction_stats': 3}
def test_health():
    with TestClient(app) as client:
        response = client.get('/health')
    assert 200 == response.status_code
    assert 'true' == response.text


def test_predict_single():
    with TestClient(app) as client:
        response = client.get('/predict', json={'data': CHECK_INSTANCE})
    assert 200 == response.status_code
    assert 1 == len(response.json())
    assert response.json()['Id0'] in [0, 1]

def test_predict_single():
    with TestClient(app) as client:
        response = client.get('/predict', json={'data': BATCH_INSTANCE})
    assert 200 == response.status_code
    assert 5 == len(response.json())
    assert set([0, 1]) == set(response.json().values())
    assert ['Id' + str(i) for i in range(5)] == list(response.json().keys())
