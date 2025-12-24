import os
from pathlib import Path
from pyspark.sql import SparkSession, functions as F

# ---- настройки подключения к PostgreSQL (env с дефолтами) ----
PG_HOST = os.getenv("PG_HOST", "postgres")   # локально можно export PG_HOST=localhost
PG_PORT = os.getenv("PG_PORT", "5435")       # для docker-compose внешний порт 5435
PG_DB   = os.getenv("PG_DB", "nyc_taxi")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "1234")

PG_URL = f"jdbc:postgresql://{PG_HOST}:{PG_PORT}/{PG_DB}"

PG_PROPS = {
    "user": PG_USER,
    "password": PG_PASSWORD,
    "driver": "org.postgresql.Driver",   # если драйвер уже добавлен
}

def main():
    spark = (
        SparkSession.builder
        .appName("nyc_taxi_build_fact_trips_agg")
        .master("local[*]")
        .config("spark.jars.packages", "org.postgresql:postgresql:42.7.4")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "0.0.0.0")
        .getOrCreate()
    )

    # 1. Читаем фичи (пытаемся взять путь из окружения, иначе рядом с проектом)
    env_root = os.getenv("PROJECT_ROOT") or os.getenv("BIGDATA_PROJECT_ROOT")
    default_root = Path(__file__).resolve().parent.parent
    project_root = Path(env_root).expanduser().resolve() if env_root else default_root
    base_data_dir = project_root / "data"
    df = spark.read.parquet(str(base_data_dir / "processed" / "nyc_taxi_features.parquet"))

    print("Schema of features:")
    df.printSchema()

    # 2. Проверяем нужные колонки
    needed_cols = [
        "pickup_date",
        "pickup_hour",
        "pickup_dow",
        "is_weekend",
        "PU_borough",
        "PU_zone",
        "trip_duration_min",
        "trip_speed_mph",
        "trip_distance",
        "total_amount",
        "is_airport_pickup",
        "is_airport_dropoff",
    ]

    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"В фичах не хватает колонок: {missing}")

    # 3. Группировка (без day_segment, потому что его нет)
    fact = (
        df.groupBy(
            "pickup_date",
            "pickup_hour",
            "pickup_dow",
            "PU_borough",
            "PU_zone",
            "is_weekend",
        )
        .agg(
            F.count("*").alias("trips_count"),
            F.round(F.avg("trip_duration_min"), 2).alias("avg_duration_min"),
            F.round(F.avg("trip_speed_mph"), 2).alias("avg_speed_mph"),
            F.round(F.avg("total_amount"), 2).alias("avg_total_amount"),
            F.round(F.sum("total_amount"), 2).alias("sum_revenue"),
            F.round(F.avg("trip_distance"), 2).alias("avg_trip_distance"),
            F.sum("is_airport_pickup").alias("airport_pickups_count"),
            F.sum("is_airport_dropoff").alias("airport_dropoffs_count"),
        )
    )

    print("Schema of fact_trips_agg:")
    fact.printSchema()

    # 4. Запись в PostgreSQL
    fact.write.mode("overwrite").jdbc(PG_URL, "fact_trips_agg", properties=PG_PROPS)
    print("Витрина fact_trips_agg обновлена в PostgreSQL")

    spark.stop()

if __name__ == "__main__":
    main()
