from pathlib import Path
from pyspark.sql import SparkSession, functions as F, Window
import os


PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
#PROJECT_ROOT = Path("/usr/local/airflow/bigdata_project")
BASE_DATA_DIR = PROJECT_ROOT / "data"

RAW_YELLOW_PATH = str(BASE_DATA_DIR / "raw" / "tlc_yellow" / "*.parquet")
TAXI_ZONE_LOOKUP_PATH = str(BASE_DATA_DIR / "reference" / "taxi_zone_lookup.csv")
OUTPUT_FEATURES_PATH = str(BASE_DATA_DIR / "processed" / "nyc_taxi_features.parquet")


def main():
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("RAW_YELLOW_PATH:", RAW_YELLOW_PATH)

    spark = (
        SparkSession.builder
        .appName("nyc_taxi_transform_features")
        .master("local[*]")
        .config("spark.driver.memory", "8g")   # подстрой под свою RAM
        .config("spark.executor.memory", "4g")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "0.0.0.0")
        .getOrCreate()
    )

    # 1. Читаем сырые данные
    df_raw = spark.read.parquet(RAW_YELLOW_PATH)
    print("Schema raw:")
    df_raw.printSchema()

    # ❗ Для отладки: берём только часть данных
    df_raw = df_raw.sample(fraction=0.1, seed=42)
    # или: df_raw = df_raw.limit(1_000_000)

    df_raw = df_raw.filter(
        F.col("tpep_dropoff_datetime") >= F.col("tpep_pickup_datetime")
    )

    # 2. duration + speed
    df_tmp = df_raw.withColumn(
        "trip_duration_min",
        (F.unix_timestamp("tpep_dropoff_datetime") - F.unix_timestamp("tpep_pickup_datetime")) / 60.0
    )

    df_tmp = df_tmp.withColumn(
        "trip_speed_mph",
        F.when(
            F.col("trip_duration_min") > 0,
            F.col("trip_distance") / (F.col("trip_duration_min") / 60.0)
        )
    )

    df_clean = df_tmp.filter(
        (F.col("trip_duration_min") >= 1) &
        (F.col("trip_duration_min") <= 180) &
        (F.col("trip_distance") >= 0.1) &
        (F.col("trip_distance") <= 80) &
        (F.col("total_amount") >= 0) &
        (F.col("total_amount") <= 500) &
        (F.col("trip_speed_mph") >= 1) &
        (F.col("trip_speed_mph") <= 80)
    )

    # 3. Зоны
    zones = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(TAXI_ZONE_LOOKUP_PATH)
    )

    zones_pickup = zones.select(
        F.col("LocationID").cast("int").alias("PULocationID_lookup"),
        F.col("Borough").alias("PU_borough"),
        F.col("Zone").alias("PU_zone"),
        F.col("service_zone").alias("PU_service_zone"),
    )

    df_enriched = df_clean.join(
        zones_pickup,
        df_clean.PULocationID == zones_pickup.PULocationID_lookup,
        "left"
    ).drop("PULocationID_lookup")

    zones_dropoff = zones.select(
        F.col("LocationID").cast("int").alias("DOLocationID_lookup"),
        F.col("Borough").alias("DO_borough"),
        F.col("Zone").alias("DO_zone"),
        F.col("service_zone").alias("DO_service_zone"),
    )

    df_enriched = df_enriched.join(
        zones_dropoff,
        df_enriched.DOLocationID == zones_dropoff.DOLocationID_lookup,
        "left"
    ).drop("DOLocationID_lookup")

    # 4. Временные фичи
    df_feat = df_enriched \
        .withColumn("pickup_ts", F.col("tpep_pickup_datetime").cast("timestamp")) \
        .withColumn("pickup_date", F.to_date("pickup_ts")) \
        .withColumn("pickup_hour", F.hour("pickup_ts")) \
        .withColumn("pickup_dow_str", F.date_format("pickup_ts", "E")) \
        .withColumn("pickup_week", F.weekofyear("pickup_ts")) \
        .withColumn(
            "is_weekend",
            F.col("pickup_dow_str").isin("Sat", "Sun").cast("int")
        ) \
        .withColumn("pickup_dow", F.dayofweek("pickup_ts"))

    df_feat = df_feat.withColumn(
        "is_airport_pickup",
        F.lower(F.col("PU_zone")).contains("airport").cast("int")
    ).withColumn(
        "is_airport_dropoff",
        F.lower(F.col("DO_zone")).contains("airport").cast("int")
    )

    # 5. Окно
    w = Window.partitionBy("PULocationID", "pickup_date", "pickup_hour")

    df_feat = df_feat \
        .withColumn("loc_hourly_pickups", F.count("*").over(w)) \
        .withColumn("loc_avg_speed_hour", F.avg("trip_speed_mph").over(w)) \
        .withColumn("loc_avg_duration_hour", F.avg("trip_duration_min").over(w))

    df_feat = df_feat \
        .withColumn("hour_sin", F.sin(2 * 3.14159265 * F.col("pickup_hour") / 24.0)) \
        .withColumn("hour_cos", F.cos(2 * 3.14159265 * F.col("pickup_hour") / 24.0))

    df_feat.printSchema()

    df_feat.write.mode("overwrite").parquet(OUTPUT_FEATURES_PATH)
    print("Features saved to", OUTPUT_FEATURES_PATH)

    spark.stop()


if __name__ == "__main__":
    main()
