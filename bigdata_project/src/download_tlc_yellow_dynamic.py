# src/download_tlc_yellow_dynamic.py

import re
from pathlib import Path
from datetime import datetime
from typing import List

import requests



# Страница с описанием датасетов TLC
TLC_PAGE_URL = "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"

# Базовый URL, откуда скачиваются parquet-файлы с поездками
BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"

# URL-ы для справочников зон
TAXI_ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
TAXI_ZONE_SHAPEFILE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"  # по желанию

# Папки для сохранения
OUT_DIR = Path("../data/raw/tlc_yellow")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REF_DIR = Path("../data/reference")
REF_DIR.mkdir(parents=True, exist_ok=True)


def get_available_year_months() -> List[str]:
    """
    Скачиваем HTML страницы TLC и вытаскиваем все ссылки вида
    yellow_tripdata_YYYY-MM.parquet.

    Возвращаем отсортированный список уникальных 'YYYY-MM' по дате.
    """
    print(f"Fetching TLC page: {TLC_PAGE_URL}")
    resp = requests.get(TLC_PAGE_URL)
    resp.raise_for_status()
    html = resp.text

    pattern = r"yellow_tripdata_(\d{4}-\d{2})\.parquet"
    ym_list = re.findall(pattern, html)

    if not ym_list:
        raise RuntimeError(
            "Не удалось найти ни одного yellow_tripdata_YYYY-MM.parquet на странице TLC"
        )

    ym_unique = sorted(
        set(ym_list),
        key=lambda s: datetime.strptime(s, "%Y-%m")
    )

    print(f"Найдено {len(ym_unique)} доступных месяцев для yellow_tripdata.")
    print("Первые несколько:", ym_unique[:5], " ... Последние несколько:", ym_unique[-5:])
    return ym_unique


def get_last_n_months(all_ym: List[str], n: int) -> List[str]:
    """
    Берёт последние n месяцев из списка 'YYYY-MM'.
    Если данных меньше, чем n — возвращает весь список.
    """
    if len(all_ym) >= n:
        last = all_ym[-n:]
        print(f"Берём последние {n} месяцев: {last}")
        return last
    else:
        print(
            f"Доступно только {len(all_ym)} месяцев (< {n}), "
            f"будем использовать все: {all_ym}"
        )
        return all_ym


def download_parquet_for_month(ym: str) -> None:
    """
    Скачивает один parquet-файл для заданного 'YYYY-MM'.
    """
    url = f"{BASE_URL}/yellow_tripdata_{ym}.parquet"
    out_path = OUT_DIR / f"yellow_tripdata_{ym}.parquet"

    if out_path.exists():
        print(f"[SKIP] Файл уже существует: {out_path}")
        return

    print(f"[DOWNLOAD] {ym}: {url} -> {out_path}")
    r = requests.get(url, stream=True)
    r.raise_for_status()

    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    print(f"[DONE] {out_path}")


def download_last_n_months(n: int = 6) -> None:
    """
    Основная функция для поездок:
    - парсит страницу TLC,
    - выбирает последние n месяцев,
    - скачивает parquet-файлы для них.
    """
    all_ym = get_available_year_months()
    last_ym = get_last_n_months(all_ym, n)

    for ym in last_ym:
        download_parquet_for_month(ym)



def download_taxi_zone_lookup() -> None:
    """
    Скачивает taxi_zone_lookup.csv (и опционально shapefile)
    в папку data/reference.
    """
    lookup_path = REF_DIR / "taxi_zone_lookup.csv"
    if lookup_path.exists():
        print(f"[SKIP] Lookup уже существует: {lookup_path}")
    else:
        print(f"[DOWNLOAD] Taxi Zone Lookup -> {lookup_path}")
        r = requests.get(TAXI_ZONE_LOOKUP_URL, stream=True)
        r.raise_for_status()
        with open(lookup_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        print(f"[DONE] {lookup_path}")

    shp_path = REF_DIR / "taxi_zones.zip"
    if shp_path.exists():
        print(f"[SKIP] Shapefile уже существует: {shp_path}")
    else:
        print(f"[DOWNLOAD] Taxi Zone Shapefile -> {shp_path}")
        r = requests.get(TAXI_ZONE_SHAPEFILE_URL, stream=True)
        r.raise_for_status()
        with open(shp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        print(f"[DONE] {shp_path}")


def main():
    # последние N месяцев поездок
    download_last_n_months(n=6)
    # подтягиваем справочник зон
    download_taxi_zone_lookup()


if __name__ == "__main__":
    main()
