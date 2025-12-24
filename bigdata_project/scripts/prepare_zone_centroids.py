from pathlib import Path

import geopandas as gpd
import pandas as pd


# ---------------------------
# Пути
# ---------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

SHAPE_PATH = BASE_DIR / "data" / "taxi_zones" / "taxi_zones.shp"
LOOKUP_PATH = BASE_DIR / "data" / "reference" / "taxi_zone_lookup.csv"
OUT_PATH = BASE_DIR / "data" / "reference" / "zone_centroids.csv"


def main():
    print(f"Читаю шейп-файл: {SHAPE_PATH}")
    gdf = gpd.read_file(SHAPE_PATH)

    print(f"Читаю lookup: {LOOKUP_PATH}")
    df_lookup = pd.read_csv(LOOKUP_PATH)

    # Обычно в этих файлах есть колонка LocationID
    # В шейпе она может называться 'LocationID' или 'location_i'
    cols_lower = {c.lower(): c for c in gdf.columns}
    loc_col = None
    for cand in ["locationid", "location_id"]:
        if cand in cols_lower:
            loc_col = cols_lower[cand]
            break
    if loc_col is None:
        raise ValueError(f"В шейпе не нашёл колонку LocationID. Колонки: {list(gdf.columns)}")

    if "LocationID" not in gdf.columns:
        gdf = gdf.rename(columns={loc_col: "LocationID"})

    # В lookup колонка обычно называется 'LocationID' и 'Zone'
    if "LocationID" not in df_lookup.columns:
        raise ValueError(f"В lookup нет LocationID. Колонки: {list(df_lookup.columns)}")
    if "Zone" not in df_lookup.columns:
        raise ValueError(f"В lookup нет Zone. Колонки: {list(df_lookup.columns)}")

    # Приводим к WGS84 (широта/долгота), если нужно
    if gdf.crs is None:
        print("CRS у шейпа не задан, предполагаю EPSG:4326 (WGS84).")
    elif gdf.crs.to_string().upper() != "EPSG:4326":
        print(f"Преобразую CRS из {gdf.crs} в EPSG:4326…")
        gdf = gdf.to_crs(epsg=4326)

    # Джойним шейп и lookup по LocationID
    gdf = gdf.merge(df_lookup[["LocationID", "Zone"]], on="LocationID", how="left")

    # Считаем центроиды
    gdf["centroid"] = gdf.geometry.centroid
    gdf["lat"] = gdf["centroid"].y
    gdf["lon"] = gdf["centroid"].x

    # Готовим финальный датафрейм
    df_out = (
        gdf[["Zone", "lat", "lon"]]
        .dropna(subset=["Zone"])
        .rename(columns={"Zone": "PU_zone"})
        .sort_values("PU_zone")
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"Сохранено {len(df_out)} зон в {OUT_PATH}")


if __name__ == "__main__":
    main()
