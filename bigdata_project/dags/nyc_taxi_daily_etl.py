import os
from datetime import datetime, timedelta
from pathlib import Path
from airflow import DAG
from airflow.operators.bash import BashOperator

# -----------------------------
# Определяем PROJECT_DIR относительно файла DAG
# -----------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent  # <- две папки вверх от DAG файла
DEFAULT_PY = PROJECT_DIR / ".venv" / "bin" / "python"
PYTHON_BIN = os.getenv("PYTHON_BIN") or (str(DEFAULT_PY) if DEFAULT_PY.exists() else "python3")

# Прокидываем окружение в таски (подхват PG и путей)
ENV_VARS = {
    "PROJECT_ROOT": str(PROJECT_DIR),
    "PG_HOST": os.getenv("PG_HOST", "localhost"),
    "PG_PORT": os.getenv("PG_PORT", "5435"),
    "PG_DB": os.getenv("PG_DB", "nyc_taxi"),
    "PG_USER": os.getenv("PG_USER", "postgres"),
    "PG_PASSWORD": os.getenv("PG_PASSWORD", "1234"),
}

DOWNLOAD_SCRIPT = PROJECT_DIR / "src/download_tlc_yellow_dynamic.py"
TRANSFORM_FEATURES_SCRIPT = PROJECT_DIR / "spark_jobs/transform_features.py"
BUILD_FACT_TRIPS_AGG_SCRIPT = PROJECT_DIR / "spark_jobs/build_fact_trips_agg.py"
BUILD_ML_DATASET_SCRIPT = PROJECT_DIR / "spark_jobs/build_ml_dataset_trips_hourly.py"

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="nyc_taxi_daily_etl",
    description="Daily ETL: download -> spark -> marts (ml_trips_hourly)",
    default_args=default_args,
    schedule_interval="0 4 * * *",
    start_date=datetime(2024, 12, 1),
    catchup=False,
    tags=["nyc_taxi", "etl", "daily"],
) as dag:

    download_task = BashOperator(
        task_id="download_raw_tlc",
        bash_command=f"{PYTHON_BIN} {DOWNLOAD_SCRIPT}",
        env=ENV_VARS,
    )

    transform_features = BashOperator(
        task_id="spark_transform_features",
        bash_command=f"{PYTHON_BIN} {TRANSFORM_FEATURES_SCRIPT}",
        env=ENV_VARS,
    )

    build_fact_trips_agg = BashOperator(
        task_id="build_fact_trips_agg",
        bash_command=f"{PYTHON_BIN} {BUILD_FACT_TRIPS_AGG_SCRIPT}",
        env=ENV_VARS,
    )

    build_ml_dataset = BashOperator(
        task_id="build_ml_trips_hourly",
        bash_command=f"{PYTHON_BIN} {BUILD_ML_DATASET_SCRIPT}",
        env=ENV_VARS,
    )

    # Задачи по порядку
    download_task >> transform_features >> build_fact_trips_agg >> build_ml_dataset
