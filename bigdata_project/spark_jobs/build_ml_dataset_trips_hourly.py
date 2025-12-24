import os
from pathlib import Path
from pyspark.sql import SparkSession, functions as F, Window

PG_HOST = os.getenv("PG_HOST", "postgres")   # локально можно PG_HOST=localhost
PG_PORT = os.getenv("PG_PORT", "5435")       # внешний порт docker-compose
PG_DB = os.getenv("PG_DB", "nyc_taxi")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "1234")

PG_URL = f"jdbc:postgresql://{PG_HOST}:{PG_PORT}/{PG_DB}"

PG_PROPS = {
    "user": PG_USER,
    "password": PG_PASSWORD,
    "driver": "org.postgresql.Driver",
}

def main():
    spark = (
        SparkSession.builder
        .appName("nyc_taxi_build_ml_trips_hourly")
        .master("local[*]")
        .config("spark.jars.packages", "org.postgresql:postgresql:42.7.4")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "0.0.0.0")
        .getOrCreate()
    )

    # 1. Читаем витрину fact_trips_agg из Postgres
    fact = (
        spark.read
        .jdbc(PG_URL, "fact_trips_agg", properties=PG_PROPS)
    )

    print("Schema fact_trips_agg:")
    fact.printSchema()

    # 2. Строим timestamp из pickup_date + pickup_hour
    fact = fact.withColumn(
        "pickup_ts",
        F.to_timestamp(
            F.concat_ws(
                " ",
                F.col("pickup_date").cast("string"),
                F.col("pickup_hour").cast("string"),
            ),
            "yyyy-MM-dd H",
        ),
    )

    # 3. Циклические признаки часа
    fact = fact.withColumn(
        "hour_sin",
        F.sin(2 * 3.14159265 * F.col("pickup_hour") / 24.0),
    ).withColumn(
        "hour_cos",
        F.cos(2 * 3.14159265 * F.col("pickup_hour") / 24.0),
    )

    # 4. Окно для временных лагов: по зоне и часу
    w_zone_hour = (
        Window
        .partitionBy("PU_borough", "PU_zone", "pickup_hour")
        .orderBy("pickup_ts")
    )

    # 5. Лаги и скользящее среднее
    fact_ml = (
        fact
        .withColumn("trips_lag_1d", F.lag("trips_count", 1).over(w_zone_hour))
        .withColumn("trips_lag_7d", F.lag("trips_count", 7).over(w_zone_hour))
        .withColumn(
            "trips_ma_7d",
            F.avg("trips_count").over(
                w_zone_hour.rowsBetween(-7, -1)  # 7 предыдущих наблюдений
            ),
        )
    )

    # 6. Фильтруем строки без достаточной истории
    fact_ml = fact_ml.filter(
        F.col("trips_lag_1d").isNotNull()
        & F.col("trips_lag_7d").isNotNull()
        & F.col("trips_ma_7d").isNotNull()
    )

    # 7. Финальный набор колонок
    ml_cols = [
        "pickup_ts",
        "pickup_date",
        "pickup_hour",
        "pickup_dow",
        "is_weekend",
        "PU_borough",
        "PU_zone",

        "trips_count",      # таргет

        "trips_lag_1d",
        "trips_lag_7d",
        "trips_ma_7d",

        "hour_sin",
        "hour_cos",
    ]

    fact_ml = fact_ml.select(*ml_cols)

    print("Schema ml_trips_hourly:")
    fact_ml.printSchema()

    # 8. Пишем в Postgres
    (
        fact_ml.write
        .mode("overwrite")
        .jdbc(PG_URL, "ml_trips_hourly", properties=PG_PROPS)
    )

    print("ML-датасет ml_trips_hourly обновлён в PostgreSQL")

    spark.stop()


if __name__ == "__main__":
    main()
