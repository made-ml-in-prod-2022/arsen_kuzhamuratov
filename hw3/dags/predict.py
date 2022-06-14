from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago
from docker.types import Mount

SAVE_DIR_GENERATE = "/data/raw/{{ ds }}"
SAVE_DIR_PROCESSED = "/data/processed/{{ ds }}"
MODEL_PATH = "/data/processed/{{ ds }}/model.pkl"
MOUNT_SOURCE = Mount(
    source="/home/arsen/Arsen/MADE-DS21/arsen_kuzhamuratov/hw3/data",
    target="/data",
    type='bind'
    )


default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        "predict",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(0),
) as dag:
    predict = DockerOperator(
        image="airflow-model-predict",
        command="--in_dir {} --model_path {} --pred_path {}/predictions.csv".format(SAVE_DIR_PROCESSED, MODEL_PATH, SAVE_DIR_PROCESSED),
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        network_mode="bridge",
        mounts=[MOUNT_SOURCE]
    )

    predict