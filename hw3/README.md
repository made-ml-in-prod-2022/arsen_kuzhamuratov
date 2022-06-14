## Project organization:
```
├── dags
│   ├── generate_data.py
│   ├── predict.py
│   └── train.py
├── data
├── docker-compose.yml
├── images
│   ├── airflow-data-generate
│   │   ├── Dockerfile
│   │   └── generate_data.py
│   ├── airflow-data-move
│   │   ├── Dockerfile
│   │   └── move_data.py
│   ├── airflow-data-split
│   │   ├── Dockerfile
│   │   └── split_data.py
│   ├── airflow-docker
│   │   └── Dockerfile
│   ├── airflow-ml-base
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── airflow-model-predict
│   │   ├── Dockerfile
│   │   └── predict.py
│   ├── airflow-model-train
│   │   ├── Dockerfile
│   │   └── train.py
│   └── airflow-model-validate
│       ├── Dockerfile
│       └── validate.py
└── README.md
```

## Configuration
Change `model_dir/` in `docker-compose.yml`
```
AIRFLOW_VAR_MODELPATH=/data/models/2022-06-14
```
## Run app
```
export FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")
sudo docker-compose up --build
```