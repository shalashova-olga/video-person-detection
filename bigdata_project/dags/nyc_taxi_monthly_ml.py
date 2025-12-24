import os
from datetime import datetime, timedelta
from pathlib import Path
from airflow import DAG
from airflow.operators.bash import BashOperator


# ==========================
#  НАСТРОЙКИ
# ==========================
PROJECT_DIR = Path(__file__).resolve().parent.parent  # <- две папки вверх от DAG файла
DEFAULT_PY = PROJECT_DIR / ".venv" / "bin" / "python"
PYTHON_BIN = os.getenv("PYTHON_BIN") or (str(DEFAULT_PY) if DEFAULT_PY.exists() else "python3")

ENV_VARS = {
    "PROJECT_ROOT": PROJECT_DIR,
    "PG_HOST": os.getenv("PG_HOST", "localhost"),
    "PG_PORT": os.getenv("PG_PORT", "5435"),
    "PG_DB": os.getenv("PG_DB", "nyc_taxi"),
    "PG_USER": os.getenv("PG_USER", "postgres"),
    "PG_PASSWORD": os.getenv("PG_PASSWORD", "1234"),
}

TRAIN_SCRIPT = "ml/train_trips_catboost_prod.py"
FORECAST_SCRIPT = "ml/predict_next_month.py"
EVAL_SCRIPT = "scripts/evaluate_model_hourly.py"

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="nyc_taxi_monthly_ml",
    description="Monthly ML: train CatBoost -> forecast -> evaluate quality",
    default_args=default_args,
    schedule_interval="0 6 2 * *",  # 2-е число месяца в 06:00
    start_date=datetime(2024, 12, 1),
    catchup=False,
    tags=["nyc_taxi", "ml", "monthly"],
) as dag:

    train_model = BashOperator(
        task_id="train_model",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            f"{PYTHON_BIN} {TRAIN_SCRIPT}"
        ),
        env=ENV_VARS,
    )

    run_monthly_forecast = BashOperator(
        task_id="run_monthly_forecast",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            f"{PYTHON_BIN} {FORECAST_SCRIPT}"
        ),
        env=ENV_VARS,
    )

    evaluate_model = BashOperator(
        task_id="evaluate_model",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            f"{PYTHON_BIN} {EVAL_SCRIPT}"
        ),
        env=ENV_VARS,
    )

    train_model >> run_monthly_forecast >> evaluate_model
