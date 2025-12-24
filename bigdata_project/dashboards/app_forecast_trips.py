import os
from datetime import date

import pandas as pd
from sqlalchemy import create_engine
import streamlit as st
import plotly.express as px

# ==========================
#   НАСТРОЙКИ ПОДКЛЮЧЕНИЯ
# ==========================
PG_USER = "postgres"
PG_PASSWORD = "1234"
PG_HOST = "localhost"
PG_PORT = 5433
PG_DB = "nyc_taxi"

CONN_STR = f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"


# ==========================
#   ЗАГРУЗКА ДАННЫХ
# ==========================

@st.cache_data(show_spinner=True)
def load_hist_data():
    """
    Исторические агрегаты по поездкам: ml_trips_hourly
    """
    engine = create_engine(CONN_STR)
    query = """
        SELECT
            pickup_ts,
            pickup_date,
            pickup_hour,
            pickup_dow,
            "PU_borough",
            "PU_zone",
            trips_count
        FROM ml_trips_hourly
    """
    df = pd.read_sql(query, engine)
    df["pickup_ts"] = pd.to_datetime(df["pickup_ts"])
    df["pickup_date"] = pd.to_datetime(df["pickup_date"]).dt.date
    return df


@st.cache_data(show_spinner=True)
def load_forecast_data():
    """
    Прогнозы из таблицы pred_trips_hourly,
    которые записывает скрипт с CatBoost.
    """
    engine = create_engine(CONN_STR)
    query = """
        SELECT
            predict_for_ts,
            predict_for_date,
            predict_for_hour,
            "PU_borough",
            "PU_zone",
            predicted_trips_count,
            prediction_run_ts,
            model_name
        FROM pred_trips_hourly
    """
    df = pd.read_sql(query, engine)
    df["predict_for_ts"] = pd.to_datetime(df["predict_for_ts"])
    df["predict_for_date"] = pd.to_datetime(df["predict_for_date"]).dt.date
    df["prediction_run_ts"] = pd.to_datetime(df["prediction_run_ts"])
    return df


@st.cache_data(show_spinner=True)
def load_zone_centroids():
    """
    Центроиды зон для карты.

    Ожидается CSV с колонками:
        PU_zone, lat, lon

    Пытаемся найти:
        data/zone_centroids.csv
        data/reference/zone_centroids.csv
    """
    candidates = [
        "data/zone_centroids.csv",
        "data/reference/zone_centroids.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            # нормализуем названия колонок
            cols = {c.lower(): c for c in df.columns}
            # приводим к нужным именам
            if "pu_zone" not in df.columns and "pu_zone" in cols:
                df.rename(columns={cols["pu_zone"]: "PU_zone"}, inplace=True)
            if "lat" not in df.columns and "latitude" in cols:
                df.rename(columns={cols["latitude"]: "lat"}, inplace=True)
            if "lon" not in df.columns and ("lng" in cols or "longitude" in cols):
                if "lng" in cols:
                    df.rename(columns={cols["lng"]: "lon"}, inplace=True)
                else:
                    df.rename(columns={cols["longitude"]: "lon"}, inplace=True)
            return df
    return pd.DataFrame()


# ==========================
#   UI / ЛОГИКА ПРИЛОЖЕНИЯ
# ==========================

def main():
    st.set_page_config(
        page_title="NYC Taxi — анализ и прогноз",
        layout="wide",
    )

    st.title("NYC Taxi — анализ спроса и прогноз")

    st.markdown(
        """
        Этот дашборд в первую очередь про **анализ данных**, а модельный прогноз —
        дополнительный слой внизу.

        **Источники:**
        * `ml_trips_hourly` — фактическое количество поездок по `(borough, zone, hour)`
        * `pred_trips_hourly` — 30-дневный прогноз от CatBoost
        """
    )

    # --- Загружаем данные ---
    try:
        df_hist = load_hist_data()
    except Exception as e:
        st.error(f"Ошибка при загрузке истории (ml_trips_hourly): {e}")
        return

    if df_hist.empty:
        st.warning("Таблица ml_trips_hourly пуста или недоступна.")
        return

    # прогноз может быть пустым — это ок, просто отключим вкладку прогноза
    try:
        df_pred = load_forecast_data()
    except Exception:
        df_pred = pd.DataFrame()

    # --- Сайдбар с общими фильтрами для анализа ---
    st.sidebar.header("Фильтры")

    # Диапазон дат
    min_date = df_hist["pickup_date"].min()
    max_date = df_hist["pickup_date"].max()

    default_start = max_date - pd.Timedelta(days=30)
    if default_start < min_date:
        default_start = min_date

    date_range = st.sidebar.date_input(
        "Диапазон дат (факт)",
        value=(default_start, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(date_range, (list, tuple)):
        start_date, end_date = date_range
    else:
        # если вдруг streamlit вернул одну дату
        start_date, end_date = min_date, max_date

    if start_date > end_date:
        st.sidebar.error("Начальная дата больше конечной. Исправь диапазон.")
        return

    # фильтруем по дате
    df_hist = df_hist[
        (df_hist["pickup_date"] >= start_date)
        & (df_hist["pickup_date"] <= end_date)
    ].copy()

    # Borough
    boroughs = sorted(df_hist["PU_borough"].dropna().unique().tolist())
    boroughs_with_all = ["Все"] + boroughs
    borough = st.sidebar.selectbox("Borough", boroughs_with_all)

    # Зоны зависят от выбранного borough
    if borough == "Все":
        zones_available = df_hist["PU_zone"].dropna().unique().tolist()
    else:
        zones_available = (
            df_hist.loc[df_hist["PU_borough"] == borough, "PU_zone"]
            .dropna()
            .unique()
            .tolist()
        )

    zones_available = sorted(zones_available)
    zones_with_all = ["Все"] + zones_available

    zone = st.sidebar.selectbox("Зона (PU_zone)", zones_with_all)

    # Час суток
    hours = sorted(df_hist["pickup_hour"].dropna().unique().tolist())
    hour = st.sidebar.selectbox("Час суток (для прогнозной вкладки)", hours)

    # доп. фильтрация для анализа (borough/zone)
    df_hist_filtered = df_hist.copy()
    if borough != "Все":
        df_hist_filtered = df_hist_filtered[df_hist_filtered["PU_borough"] == borough]
    if zone != "Все":
        df_hist_filtered = df_hist_filtered[df_hist_filtered["PU_zone"] == zone]

    # ==========================
    #   ТАБЫ: Аналитика / Карта / Прогноз
    # ==========================
    # Если прогноз пуст, всё равно создадим вкладку, но подсветим, что данных нет.
    tab_analysis, tab_map, tab_forecast = st.tabs(
        ["📊 Аналитика спроса", "🗺 Карта зон", "🤖 Прогноз (CatBoost)"]
    )

    # ---------------------------------
    #   ТАБ 1: АНАЛИТИКА СПРОСА
    # ---------------------------------
    with tab_analysis:
        st.subheader("Аналитика спроса (фактические данные)")

        if df_hist_filtered.empty:
            st.warning("Нет данных для выбранных фильтров.")
        else:
            # Метрики
            total_trips = float(df_hist_filtered["trips_count"].sum())
            avg_trips_per_hour = float(
                df_hist_filtered.groupby(["pickup_date", "pickup_hour"])["trips_count"]
                .sum()
                .mean()
            )

            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Суммарное количество поездок (факт)",
                f"{total_trips:,.0f}".replace(",", " "),
            )
            col2.metric(
                "Среднее количество поездок на час",
                f"{avg_trips_per_hour:,.1f}",
            )

            unique_zones = df_hist_filtered["PU_zone"].nunique()
            col3.metric(
                "Количество уникальных зон в выборке",
                f"{unique_zones}",
            )

            # Таймсерия по датам (сумма по всем часам)
            st.markdown("### Динамика по дням (сумма по всем часам)")

            daily = (
                df_hist_filtered.groupby("pickup_date")["trips_count"]
                .sum()
                .reset_index()
                .sort_values("pickup_date")
            )

            fig_daily = px.line(
                daily,
                x="pickup_date",
                y="trips_count",
                labels={"pickup_date": "Дата", "trips_count": "Количество поездок"},
                title="Количество поездок по дням",
            )
            st.plotly_chart(fig_daily, use_container_width=True)

            # Распределение по дням недели
            st.markdown("### Распределение спроса по дням недели")

            df_hist_filtered["pickup_dow"] = df_hist_filtered["pickup_dow"].fillna(0).astype(int)

            dow_map = {
                1: "Sun",
                2: "Mon",
                3: "Tue",
                4: "Wed",
                5: "Thu",
                6: "Fri",
                7: "Sat",
            }
            df_hist_filtered["dow_name"] = df_hist_filtered["pickup_dow"].map(dow_map)

            dow_agg = (
                df_hist_filtered.groupby("dow_name")["trips_count"]
                .sum()
                .reindex(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
                .reset_index()
                .dropna()
            )

            fig_dow = px.bar(
                dow_agg,
                x="dow_name",
                y="trips_count",
                labels={"dow_name": "День недели", "trips_count": "Количество поездок"},
                title="Спрос по дням недели",
            )
            st.plotly_chart(fig_dow, use_container_width=True)

            # Тепловая карта час × день недели
            st.markdown("### Тепловая карта: день недели × час")

            heat = (
                df_hist_filtered.groupby(["pickup_dow", "pickup_hour"])["trips_count"]
                .sum()
                .reset_index()
            )
            heat["dow_name"] = heat["pickup_dow"].map(dow_map)

            fig_heat = px.density_heatmap(
                heat,
                x="pickup_hour",
                y="dow_name",
                z="trips_count",
                histfunc="sum",
                labels={
                    "pickup_hour": "Час суток",
                    "dow_name": "День недели",
                    "trips_count": "Количество поездок",
                },
                title="Тепловая карта спроса (час × день недели)",
            )
            st.plotly_chart(fig_heat, use_container_width=True)

    # ---------------------------------
    #   ТАБ 2: КАРТА ЗОН
    # ---------------------------------
    with tab_map:
        st.subheader("Карта зон (агрегация спроса по PU_zone)")

        centroids = load_zone_centroids()
        if centroids.empty:
            st.info(
                "Файл с координатами зон не найден. "
                "Создай `data/zone_centroids.csv` с колонками `PU_zone, lat, lon`, "
                "и карта заработает."
            )
        else:
            # агрегируем спрос по зонам за выбранный период и фильтры
            zone_agg = (
                df_hist_filtered.groupby("PU_zone")["trips_count"]
                .sum()
                .reset_index()
            )

            # джойним к координатам
            map_df = zone_agg.merge(centroids, on="PU_zone", how="inner")
            if map_df.empty:
                st.warning("Не удалось сопоставить зоны с координатами. Проверь названия `PU_zone`.")
            else:
                st.markdown(
                    f"Показано зон: **{len(map_df)}**. Радиусы точек пропорциональны количеству поездок."
                )

                fig_map = px.scatter_mapbox(
                    map_df,
                    lat="lat",
                    lon="lon",
                    size="trips_count",
                    color="trips_count",
                    hover_name="PU_zone",
                    hover_data={"trips_count": True, "lat": False, "lon": False},
                    zoom=9,
                    height=600,
                )
                fig_map.update_layout(
                    mapbox_style="carto-positron",
                    margin={"r": 0, "t": 40, "l": 0, "b": 0},
                    title="Спрос по зонам (сумма поездок за выбранный период)",
                )
                st.plotly_chart(fig_map, use_container_width=True)

    # ---------------------------------
    #   ТАБ 3: ПРОГНОЗ МОДЕЛИ
    # ---------------------------------
    with tab_forecast:
        st.subheader("Прогноз CatBoost (нижний приоритет, просто как модельное дополнение)")

        if df_pred.empty:
            st.info(
                "Таблица `pred_trips_hourly` пуста или недоступна. "
                "Запусти скрипт с CatBoost-прогнозом, чтобы увидеть эту вкладку в действии."
            )
            return

        # фильтруем историю и прогноз под конкретную зону/borough/час
        hist_filt = df_hist.copy()
        if borough != "Все":
            hist_filt = hist_filt[hist_filt["PU_borough"] == borough]
        if zone != "Все":
            hist_filt = hist_filt[hist_filt["PU_zone"] == zone]

        hist_filt = hist_filt[hist_filt["pickup_hour"] == hour].copy()
        hist_filt = hist_filt.sort_values("pickup_ts")

        pred_filt = df_pred.copy()
        if borough != "Все":
            pred_filt = pred_filt[pred_filt["PU_borough"] == borough]
        if zone != "Все":
            pred_filt = pred_filt[pred_filt["PU_zone"] == zone]
        pred_filt = pred_filt[pred_filt["predict_for_hour"] == hour].copy()
        pred_filt = pred_filt.sort_values("predict_for_ts")

        if hist_filt.empty and pred_filt.empty:
            st.warning("Для выбранных параметров нет ни факта, ни прогноза.")
            return

        # приводим к общей структуре для графика
        hist_plot = hist_filt.copy()
        hist_plot["kind"] = "fact"
        hist_plot = hist_plot.rename(
            columns={"pickup_ts": "ts", "trips_count": "value"}
        )[["ts", "value", "kind"]]

        pred_plot = pred_filt.copy()
        pred_plot["kind"] = "forecast"
        pred_plot = pred_plot.rename(
            columns={"predict_for_ts": "ts", "predicted_trips_count": "value"}
        )[["ts", "value", "kind"]]

        df_plot = pd.concat([hist_plot, pred_plot], ignore_index=True)

        subtitle_boro = borough if borough != "Все" else "Все borough"
        subtitle_zone = zone if zone != "Все" else "Все зоны"

        st.markdown(f"**{subtitle_boro} / {subtitle_zone}, час={hour}: факт vs прогноз**")

        fig = px.line(
            df_plot,
            x="ts",
            y="value",
            color="kind",
            title=f"Факт vs прогноз — {subtitle_boro} / {subtitle_zone}, час={hour}",
            labels={
                "ts": "Время",
                "value": "Количество поездок",
                "kind": "Ряд",
            },
        )
        fig.update_layout(legend_title_text="Тип ряда")
        st.plotly_chart(fig, use_container_width=True)

        # Метрики по прогнозу (если он есть)
        if not pred_filt.empty:
            col1, col2, col3 = st.columns(3)

            total_future = float(pred_filt["predicted_trips_count"].sum())
            first_future_date = min(pred_filt["predict_for_date"])
            last_future_date = max(pred_filt["predict_for_date"])

            col1.metric(
                "Суммарный прогноз (по выбранному часу и зоне/зонам)",
                f"{total_future:,.0f} поездок".replace(",", " "),
                help=f"Период прогноза: {first_future_date} — {last_future_date}",
            )

            days_count = len(set(pred_filt["predict_for_date"]))
            if days_count > 0:
                avg_per_day = total_future / days_count
                col2.metric(
                    "Средний прогноз в день (этот час)",
                    f"{avg_per_day:,.1f} поездок/день",
                )

            last_run_ts = pred_filt["prediction_run_ts"].max()
            model_name = (
                pred_filt["model_name"].iloc[0]
                if "model_name" in pred_filt.columns and not pred_filt.empty
                else "N/A"
            )
            col3.write("**Инфо о модельном запуске**")
            col3.write(f"- Модель: `{model_name}`")
            col3.write(f"- Последний прогон: `{last_run_ts}`")


if __name__ == "__main__":
    main()
