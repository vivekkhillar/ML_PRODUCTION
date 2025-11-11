from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import subprocess
import os

DEFAULT_ARGS = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

DATA_PATH = "/opt/airflow/data/customer_churn.csv"
TRAIN_SCRIPT = "/opt/airflow/apps/train.py"

def load_data():
    print("📌 Checking file path:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Data loaded successfully with shape: {df.shape}")

def train_model():
    print("🚀 Training model started...")
    result = subprocess.run(["python", TRAIN_SCRIPT], capture_output=True, text=True)
    print(result.stdout)
    print("✅ Training complete!")

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

    load_task >> train_task
