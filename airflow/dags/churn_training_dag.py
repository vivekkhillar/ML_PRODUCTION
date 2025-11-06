from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import subprocess

DEFAULT_ARGS = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

DATA_PATH = "/opt/airflow/data/customer_churn.csv"
TRAIN_SCRIPT = "/opt/airflow/app/train.py"  #  Updated path

# Task 1: Load Data
def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Data loaded successfully: {df.shape}")

# Task 2: Train Model using your same script
def train_model():
    print("Training model...")
    subprocess.run(["python", TRAIN_SCRIPT], check=True)
    print(" Model training complete!")

with DAG(
    dag_id="churn_data_pipeline",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2025, 11, 6),
    schedule_interval="@daily",
    catchup=False,
) as dag:

    load_task = PythonOperator(
        task_id="load_customer_data",
        python_callable=load_data,
    )

    train_task = PythonOperator(
        task_id="train_churn_model",
        python_callable=train_model,
    )

    #   Pipeline flow
    load_task >> train_task
