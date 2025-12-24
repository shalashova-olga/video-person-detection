import os
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from catboost import CatBoostRegressor, Pool

# ---------- НАСТРОЙКИ ПОДКЛЮЧЕНИЯ К БД ----------
PG_URL_SQLA = "postgresql+psycopg2://postgres:1234@localhost:5433/nyc_taxi"

# Куда сохраняем модель
MODEL_PATH = "models/catboost_trips_hourly_optuna.cbm"

# Окно истории для обучения (в днях)
# 👉 Поставь 30, если хочешь использовать только последний месяц
WINDOW_DAYS = 90

# Гиперпараметры модели (пример)
# 👉 Подставь сюда свои значения из Optuna, если они другие
BEST_PARAMS = dict(
    loss_function="RMSE",
    depth=8,
    learning_rate=0.05,
    l2_leaf_reg=3.0,
    bagging_temperature=1.0,
    random_strength=1.0,
    border_count=128,
    iterations=1000,
    random_seed=42,
    thread_count=-1,
    od_type="Iter",
    od_wait=50,
    verbose=100,
)


def load_raw_ml_data() -> pd.DataFrame:
    """
    Загружаем агрегаты из ml_trips_hourly.
    """
    engine = create_engine(PG_URL_SQLA)

    df = pd.read_sql(
        """
        SELECT
            pickup_ts,
            pickup_date,
            pickup_hour,
            pickup_dow,
            is_weekend,
            "PU_borough",
            "PU_zone",
            trips_count
        FROM ml_trips_hourly
        """,
        engine,
    )

    df["pickup_ts"] = pd.to_datetime(df["pickup_ts"])
    df["pickup_date"] = pd.to_datetime(df["pickup_date"])
    df = df.sort_values("pickup_ts").reset_index(drop=True)

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляем time-based фичи:
      - hour_sin / hour_cos
      - лаги trips_lag_1d, trips_lag_7d
      - скользящее среднее trips_ma_7d

    Лаги считаем по каждой группе (PU_borough, PU_zone, pickup_hour),
    чтобы было консистентно с тем, как в скрипте прогноза.
    """

    df = df.copy()

    # hour_sin / hour_cos
    df["hour_sin"] = np.sin(2 * np.pi * df["pickup_hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["pickup_hour"] / 24.0)

    # Лаги и MA считаем по группам
    df = df.sort_values("pickup_ts")

    def _add_lags(group: pd.DataFrame) -> pd.DataFrame:
        g = group.sort_values("pickup_ts").copy()

        # лаг 1 день (предыдущее наблюдение по этому же часу/зоне)
        g["trips_lag_1d"] = g["trips_count"].shift(1)

        # лаг 7 дней (наблюдение 7 шагов назад)
        g["trips_lag_7d"] = g["trips_count"].shift(7)

        # среднее за последние 7 дней (по предыдущим значениям)
        g["trips_ma_7d"] = (
            g["trips_count"]
            .rolling(window=7, min_periods=7)
            .mean()
            .shift(1)
        )

        return g

    df = (
        df.groupby(["PU_borough", "PU_zone", "pickup_hour"], group_keys=False)
          .apply(_add_lags)
    )

    # отбрасываем строки, где нет лагов (первые дни каждой группы)
    df = df.dropna(
        subset=["trips_lag_1d", "trips_lag_7d", "trips_ma_7d"]
    ).reset_index(drop=True)

    return df


def train_val_split(df: pd.DataFrame, val_days: int = 7):
    """
    Делим по дате: последние val_days дней -> валидация.
    """
    max_date = df["pickup_date"].max()
    cutoff = max_date - pd.Timedelta(days=val_days)

    train_df = df[df["pickup_date"] <= cutoff].copy()
    val_df = df[df["pickup_date"] > cutoff].copy()

    return train_df, val_df


def build_pools(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """
    Собираем CatBoost Pool'ы.
    """
    feature_cols = [
        "pickup_hour",
        "pickup_dow",
        "is_weekend",
        "trips_lag_1d",
        "trips_lag_7d",
        "trips_ma_7d",
        "hour_sin",
        "hour_cos",
        "PU_borough",
        "PU_zone",
    ]
    target_col = "trips_count"

    cat_features = ["PU_borough", "PU_zone"]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)

    return train_pool, val_pool, feature_cols


def main():
    print("=== Загрузка данных из ml_trips_hourly ===")
    df = load_raw_ml_data()
    print(f"Всего строк в ml_trips_hourly: {len(df)}")

    # --- Ограничиваемся последним окном истории по датам ---
    global WINDOW_DAYS
    max_date = df["pickup_date"].max()
    cutoff_date = max_date - pd.Timedelta(days=WINDOW_DAYS)

    df = df[df["pickup_date"] >= cutoff_date].copy()

    print(
        f"Берём только последние {WINDOW_DAYS} дней: "
        f"{df['pickup_date'].min().date()} — {df['pickup_date'].max().date()}, "
        f"строк: {len(df)}"
    )
    # --------------------------------------------------------

    if len(df) < 1000:
        print("Слишком мало данных для обучения модели, выходим.")
        return

    print("=== Добавляем временные фичи и лаги ===")
    df_fe = add_time_features(df)
    print(f"После добавления лагов и фильтрации: {len(df_fe)} строк")

    print("=== Train/Val split по дате ===")
    train_df, val_df = train_val_split(df_fe, val_days=7)
    print(f"Train: {len(train_df)} строк, Val: {len(val_df)} строк")

    if len(train_df) == 0 or len(val_df) == 0:
        print("Невозможно разделить на train/val, проверь диапазон дат.")
        return

    train_pool, val_pool, feature_cols = build_pools(train_df, val_df)

    print("=== Обучение CatBoost (прод-режим, без Optuna) ===")
    print("Фичи:", feature_cols)

    model = CatBoostRegressor(**BEST_PARAMS)
    model.fit(train_pool, eval_set=val_pool)

    # Простые метрики на валидации
    val_pred = model.predict(val_pool)
    y_val = val_df["trips_count"].values

    mae = float(np.mean(np.abs(y_val - val_pred)))
    rmse = float(np.sqrt(np.mean((y_val - val_pred) ** 2)))

    print(f"Val MAE  = {mae:.3f}")
    print(f"Val RMSE = {rmse:.3f}")

    # Сохраняем модель
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save_model(MODEL_PATH)

    print(f"Модель сохранена в: {MODEL_PATH}")
    print(f"Время: {datetime.utcnow()} UTC")


if __name__ == "__main__":
    main()
