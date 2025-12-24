import os
from datetime import datetime, timedelta, time

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from catboost import CatBoostRegressor

PG_URL_SQLA = "postgresql+psycopg2://postgres:1234@localhost:5433/nyc_taxi"

MODEL_PATH = "models/catboost_trips_hourly_optuna.cbm"
PRED_TABLE = "pred_trips_hourly"

# сколько дней вперёд прогнозируем
FUTURE_DAYS = 30

VERBOSE = False


def spark_dayofweek_from_py(dt: datetime) -> int:
    """
    Spark dayofweek: 1=Sunday ... 7=Saturday
    Python weekday(): 0=Monday ... 6=Sunday

    Маппинг: Monday(0)->2, ..., Saturday(5)->7, Sunday(6)->1
    """
    return ((dt.weekday() + 1) % 7) + 1


def build_future_for_group(history: pd.DataFrame, model: CatBoostRegressor) -> pd.DataFrame:
    """
    Строим прогноз для одной группы (PU_borough, PU_zone, pickup_hour).

    history: датафрейм с колонками:
        - pickup_ts (datetime)
        - pickup_date (date)
        - pickup_hour (int)
        - pickup_dow (int)
        - is_weekend (int)
        - trips_count (float/int)
        - PU_borough (str)
        - PU_zone (str)
    """

    history = history.sort_values("pickup_ts").reset_index(drop=True)

    # если меньше 7 точек истории — лаги 7д не посчитать, пропускаем
    if len(history) < 7:
        return pd.DataFrame()

    # Берём последние 7 фактических trips_count как "исторический буфер"
    last_7 = history["trips_count"].tail(7).tolist()

    # Базовая дата — последняя доступная дата в истории для этой зоны и часа
    last_ts = history["pickup_ts"].max()
    last_date = last_ts.date()
    hour = int(history["pickup_hour"].iloc[0])

    pu_borough = history["PU_borough"].iloc[0]
    pu_zone = history["PU_zone"].iloc[0]

    future_rows = []

    for day_offset in range(1, FUTURE_DAYS + 1):
        new_date = last_date + timedelta(days=day_offset)
        new_ts = datetime.combine(new_date, time(hour=hour))

        # Временные фичи
        pickup_hour = hour
        spark_dow = spark_dayofweek_from_py(new_ts)
        is_weekend = 1 if spark_dow in (1, 7) else 0

        hour_sin = np.sin(2 * np.pi * pickup_hour / 24.0)
        hour_cos = np.cos(2 * np.pi * pickup_hour / 24.0)

        # Лаги по истории (включая уже предсказанные значения)
        trips_lag_1d = last_7[-1]
        trips_lag_7d = last_7[0]
        trips_ma_7d = float(np.mean(last_7))

        # Собираем фичи в том же формате, что и при обучении
        feature_row = {
            "pickup_hour": pickup_hour,
            "pickup_dow": spark_dow,
            "is_weekend": is_weekend,
            "trips_lag_1d": trips_lag_1d,
            "trips_lag_7d": trips_lag_7d,
            "trips_ma_7d": trips_ma_7d,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "PU_borough": pu_borough,
            "PU_zone": pu_zone,
        }

        X_row = pd.DataFrame([feature_row])

        # предсказываем
        pred_trips = float(model.predict(X_row)[0])

        # не даём уходить в отрицательные значения
        pred_trips = max(pred_trips, 0.0)

        future_rows.append(
            {
                "predict_for_ts": new_ts,
                "predict_for_date": new_date,
                "predict_for_hour": pickup_hour,
                "pickup_dow": spark_dow,
                "is_weekend": is_weekend,
                "PU_borough": pu_borough,
                "PU_zone": pu_zone,
                "predicted_trips_count": pred_trips,
                "trips_lag_1d": trips_lag_1d,
                "trips_lag_7d": trips_lag_7d,
                "trips_ma_7d": trips_ma_7d,
                "hour_sin": hour_sin,
                "hour_cos": hour_cos,
            }
        )

        # обновляем буфер last_7 с учётом нового прогноза
        last_7 = last_7[1:] + [pred_trips]

    return pd.DataFrame(future_rows)


def main():
    engine = create_engine(PG_URL_SQLA)

    # 1. Загружаем последние исторические данные из ml_trips_hourly
    #    Берём всё, но при желании можно ограничить, например, последними 90 днями.
    df_hist = pd.read_sql(
        """
        SELECT
            pickup_ts,
            pickup_date,
            pickup_hour,
            pickup_dow,
            is_weekend,
            "PU_borough" AS "PU_borough",
            "PU_zone"    AS "PU_zone",
            trips_count
        FROM ml_trips_hourly
        """,
        engine,
    )

    df_hist["pickup_ts"] = pd.to_datetime(df_hist["pickup_ts"])
    df_hist["pickup_date"] = pd.to_datetime(df_hist["pickup_date"]).dt.date

    df_hist = df_hist.sort_values("pickup_ts").reset_index(drop=True)

    print("Исторических строк в ml_trips_hourly:", len(df_hist))

    # 2. Загружаем модель CatBoost
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Не найден файл модели: {MODEL_PATH}")

    model = CatBoostRegressor()
    model.load_model(MODEL_PATH)
    print("Модель загружена из:", MODEL_PATH)

    # 3. Группируем историю по (PU_borough, PU_zone, pickup_hour)
    groups = df_hist.groupby(["PU_borough", "PU_zone", "pickup_hour"])

    all_future = []
    total_groups = 0
    used_groups = 0

    for (borough, zone, hour), hist_group in groups:
        total_groups += 1

        if VERBOSE:
            print(
                f"→ Прогноз для зоны: {borough} / {zone}, "
                f"час={hour}, историй={len(hist_group)}"
            )

        df_future_group = build_future_for_group(hist_group, model)
        if not df_future_group.empty:
            used_groups += 1
            all_future.append(df_future_group)

    # краткое резюме вместо огромной простыни
    print(f"Групп (boro/zone/hour) всего: {total_groups}")
    print(f"Где смогли построить прогноз (есть ≥7 точек истории): {used_groups}")

    if not all_future:
        print("Не удалось построить ни одного прогноза (слишком мало истории?).")
        return

    future_df = pd.concat(all_future, ignore_index=True)

    # 4. Добавляем метаданные прогноза
    prediction_run_ts = datetime.utcnow()
    future_df["prediction_run_ts"] = prediction_run_ts
    future_df["model_name"] = "catboost_trips_hourly_optuna"

    print("Всего строк прогноза:", len(future_df))

    # 5. Пишем в Postgres (перезаписываем таблицу предсказаний)
    #    Если хочешь накапливать несколько запусков, можно поменять на if_exists="append".
    future_df.to_sql(
        PRED_TABLE,
        engine,
        if_exists="replace",   # или "append" если хочешь историю запусков
        index=False,
    )

    print(f"Прогнозы записаны в таблицу {PRED_TABLE} в базе nyc_taxi")


if __name__ == "__main__":
    main()
