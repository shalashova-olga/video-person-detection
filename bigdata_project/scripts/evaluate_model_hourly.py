import numpy as np
import pandas as pd
from sqlalchemy import create_engine

PG_URL_SQLA = "postgresql+psycopg2://postgres:1234@localhost:5433/nyc_taxi"
OUT_TABLE = "model_errors_hourly"


def main():
    engine = create_engine(PG_URL_SQLA)

    # Берём факт + прогноз для последнего запуска модели
    query = """
    WITH last_run AS (
        SELECT MAX(prediction_run_ts) AS ts
        FROM pred_trips_hourly
    )
    SELECT
        f.pickup_ts,
        f.pickup_date,
        f.pickup_hour,
        f.pickup_dow,
        f."PU_borough",
        f."PU_zone",
        f.trips_count,
        p.predicted_trips_count,
        p.prediction_run_ts,
        p.model_name
    FROM ml_trips_hourly f
    JOIN pred_trips_hourly p
      ON f."PU_borough"    = p."PU_borough"
     AND f."PU_zone"       = p."PU_zone"
     AND f.pickup_hour     = p.predict_for_hour
     AND f.pickup_date     = p.predict_for_date
    JOIN last_run lr
      ON p.prediction_run_ts = lr.ts
    ;
    """

    df = pd.read_sql(query, engine)

    if df.empty:
        print("Нет пар факт+прогноз. Видимо, ещё не наступил период, для которого есть прогноз.")
        return

    # Считаем ошибки
    df["error"] = df["trips_count"] - df["predicted_trips_count"]
    df["abs_error"] = df["error"].abs()
    df["sq_error"] = df["error"] ** 2

    # MAE и RMSE по всему набору
    mae = df["abs_error"].mean()
    rmse = np.sqrt(df["sq_error"].mean())

    print(f"Всего пар факт+прогноз: {len(df)}")
    print(f"MAE  (средняя абсолютная ошибка): {mae:.3f}")
    print(f"RMSE (корень из средней квадратичной ошибки): {rmse:.3f}")

    # Можно добавить агрегаты по borough / zone / hour
    agg = (
        df.groupby(['PU_borough', 'PU_zone'])[["abs_error", "sq_error"]]
        .mean()
        .reset_index()
    )
    agg["rmse_zone"] = np.sqrt(agg["sq_error"])
    agg = agg.sort_values("rmse_zone", ascending=False)

    print("\nТоп-10 зон по RMSE (где модель промахивается сильнее всего):")
    print(agg.head(10)[["PU_borough", "PU_zone", "rmse_zone"]])

    # Пишем подробную таблицу ошибок в БД
    df.to_sql(
        OUT_TABLE,
        engine,
        if_exists="replace",   # можно поменять на 'append', если захочешь историю запусков
        index=False,
    )
    print(f"\nТаблица с ошибками записана в {OUT_TABLE} в базе nyc_taxi.")


if __name__ == "__main__":
    main()
