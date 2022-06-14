from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount


SAVE_DIR_GENERATE = "/data/raw/{{ ds }}"
SAVE_DIR_PROCESSED = "/data/processed/{{ ds }}"
MODEL_SAVE_DIR = "/data/models/{{ ds }}"
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
        "train",
        default_args=default_args,
        schedule_interval="@weekly",
        start_date=days_ago(0),
) as dag:

    move_data = DockerOperator(
        image="airflow-data-move",
        command="--in_dir {} --out_dir {}".format(SAVE_DIR_GENERATE, SAVE_DIR_PROCESSED),
        task_id="docker-airflow-move",
        do_xcom_push=False,
        network_mode="bridge",
        mounts=[MOUNT_SOURCE]
    )

    split_data = DockerOperator(
        image="airflow-data-split",
        command="--in_dir {}".format(SAVE_DIR_PROCESSED),
        task_id="docker-airflow-split",
        do_xcom_push=False,
        network_mode="bridge",
        mounts=[MOUNT_SOURCE]
    )

    train_model = DockerOperator(
        image="airflow-model-train",
        command="--in_dir {} --out_dir {}".format(SAVE_DIR_PROCESSED, MODEL_SAVE_DIR),
        task_id="docker-airflow-train",
        do_xcom_push=False,
        network_mode="bridge",
        mounts=[MOUNT_SOURCE]
    )

    val_model = DockerOperator(
        image="airflow-model-validate",
        command="--model_dir {} --data_dir {}".format(MODEL_SAVE_DIR, SAVE_DIR_PROCESSED),
        task_id="docker-airflow-val",
        do_xcom_push=False,
        network_mode="bridge",
        mounts=[MOUNT_SOURCE]
    )

    move_data >> split_data >> train_model >> val_model
