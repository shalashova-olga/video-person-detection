import numpy as np
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
def load_errors_data():
    """
    Таблица с ошибками модели, которую пишет scripts/evaluate_model_hourly.py

    Ожидаемые колонки:
        pickup_ts, pickup_date, pickup_hour, pickup_dow,
        PU_borough, PU_zone,
        trips_count, predicted_trips_count,
        prediction_run_ts, model_name,
        error, abs_error, sq_error
    """
    engine = create_engine(CONN_STR)
    df = pd.read_sql("SELECT * FROM model_errors_hourly", engine)
    df["pickup_ts"] = pd.to_datetime(df["pickup_ts"])
    df["pickup_date"] = pd.to_datetime(df["pickup_date"]).dt.date
    df["prediction_run_ts"] = pd.to_datetime(df["prediction_run_ts"])
    return df


# ==========================
#   UI / ЛОГИКА ПРИЛОЖЕНИЯ
# ==========================

def main():
    st.set_page_config(
        page_title="NYC Taxi — качество модели",
        layout="wide",
    )

    st.title("NYC Taxi — качество прогноза CatBoost")

    st.markdown(
        """
        Этот дашборд показывает, **насколько модель попадает в реальность**.

        Источник данных — таблица `model_errors_hourly`, которую заполняет
        скрипт `scripts/evaluate_model_hourly.py`.

        В таблице есть:
        * факт: `trips_count`
        * прогноз: `predicted_trips_count`
        * ошибка: `error = факт − прогноз`
        * `abs_error`, `sq_error`
        """
    )

    # --- загружаем данные ---
    try:
        df = load_errors_data()
    except Exception as e:
        st.error(
            "Не удалось загрузить таблицу `model_errors_hourly`.\n"
            "Сначала запусти `python scripts/evaluate_model_hourly.py`.\n\n"
            f"Текст ошибки: {e}"
        )
        return

    if df.empty:
        st.warning("Таблица `model_errors_hourly` пуста.")
        return

    # ==========================
    #   ФИЛЬТРЫ
    # ==========================
    st.sidebar.header("Фильтры")

    # Диапазон дат по факту
    min_date = df["pickup_date"].min()
    max_date = df["pickup_date"].max()

    default_start = max_date - pd.Timedelta(days=14)
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
        start_date, end_date = min_date, max_date

    if start_date > end_date:
        st.sidebar.error("Начальная дата больше конечной.")
        return

    df_filt = df[
        (df["pickup_date"] >= start_date)
        & (df["pickup_date"] <= end_date)
    ].copy()

    # Borough
    boroughs = sorted(df_filt["PU_borough"].dropna().unique().tolist())
    boroughs_with_all = ["Все"] + boroughs
    borough = st.sidebar.selectbox("Borough", boroughs_with_all)

    if borough != "Все":
        df_filt = df_filt[df_filt["PU_borough"] == borough]

    # Zone
    zones = sorted(df_filt["PU_zone"].dropna().unique().tolist())
    zones_with_all = ["Все"] + zones
    zone = st.sidebar.selectbox("Зона (PU_zone)", zones_with_all)

    if zone != "Все":
        df_filt = df_filt[df_filt["PU_zone"] == zone]

    # Час (опциональный фильтр)
    hours = sorted(df_filt["pickup_hour"].unique().tolist())
    hours_with_all = ["Все"] + hours
    hour = st.sidebar.selectbox("Час суток", hours_with_all)

    if hour != "Все":
        df_filt = df_filt[df_filt["pickup_hour"] == hour]

    # Выбор запуска модели (если их несколько)
    run_times = sorted(df_filt["prediction_run_ts"].dropna().unique())
    if run_times:
        latest_run = run_times[-1]
        run = st.sidebar.selectbox(
            "Запуск модели (prediction_run_ts)",
            options=["Последний"] + [str(rt) for rt in run_times],
            index=0,
        )
        if run == "Последний":
            df_filt = df_filt[df_filt["prediction_run_ts"] == latest_run]
        else:
            df_filt = df_filt[df_filt["prediction_run_ts"] == pd.to_datetime(run)]

    if df_filt.empty:
        st.warning("Нет данных под выбранные фильтры.")
        return

    # ==========================
    #   ГЛОБАЛЬНЫЕ МЕТРИКИ
    # ==========================
    st.subheader("Глобальные метрики ошибки")

    mae = df_filt["abs_error"].mean()
    rmse = np.sqrt(df_filt["sq_error"].mean())
    mape = (df_filt["abs_error"] / df_filt["trips_count"].replace(0, np.nan)).mean() * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE (средняя абсолютная ошибка)", f"{mae:.3f}")
    col2.metric("RMSE (квадратичная ошибка)", f"{rmse:.3f}")
    col3.metric("MAPE (%)", f"{mape:.2f}" if not np.isnan(mape) else "N/A")

    st.caption(
        "MAE — средняя величина промаха модели в поездках.\n"
        "RMSE сильнее наказывает большие промахи.\n"
        "MAPE — средний %% ошибки (осторожно, чувствительность к малым значениям факта)."
    )

    # ==========================
    #   АНАЛИТИКА ПО ЗОНАМ
    # ==========================
    st.markdown("## Ошибка по зонам")

    zone_agg = (
        df_filt.groupby(["PU_borough", "PU_zone"])
        .agg(
            mae=("abs_error", "mean"),
            rmse=("sq_error", lambda x: np.sqrt(x.mean())),
            mean_trips=("trips_count", "mean"),
            n=("trips_count", "count"),
        )
        .reset_index()
    )

    zone_agg = zone_agg.sort_values("rmse", ascending=False)

    st.markdown("Топ-20 зон по RMSE (где модель ошибается сильнее всего):")
    st.dataframe(
        zone_agg.head(20),
        use_container_width=True,
    )

    # ==========================
    #   ГРАФИК: ФАКТ VS ПРОГНОЗ (ВРЕМЕННОЙ РЯД)
    # ==========================
    st.markdown("## Временной ряд: факт vs прогноз")

    ts_df = df_filt.sort_values("pickup_ts").copy()
    ts_plot = pd.concat(
        [
            ts_df.assign(kind="fact", value=ts_df["trips_count"])[["pickup_ts", "value", "kind"]],
            ts_df.assign(kind="forecast", value=ts_df["predicted_trips_count"])[["pickup_ts", "value", "kind"]],
        ],
        ignore_index=True,
    )

    fig_ts = px.line(
        ts_plot,
        x="pickup_ts",
        y="value",
        color="kind",
        labels={"pickup_ts": "Время", "value": "Количество поездок", "kind": "Ряд"},
        title="Факт vs прогноз во времени (после фильтров)",
    )
    fig_ts.update_layout(legend_title_text="Тип ряда")
    st.plotly_chart(fig_ts, use_container_width=True)

    # ==========================
    #   ТЕПЛОВАЯ КАРТА ОШИБОК
    # ==========================
    st.markdown("## Тепловая карта ошибки (день недели × час)")

    dow_map = {
        1: "Sun",
        2: "Mon",
        3: "Tue",
        4: "Wed",
        5: "Thu",
        6: "Fri",
        7: "Sat",
    }
    df_filt["dow_name"] = df_filt["pickup_dow"].map(dow_map)

    heat = (
        df_filt.groupby(["dow_name", "pickup_hour"])["abs_error"]
        .mean()
        .reset_index()
    )

    # сортируем дни недели по порядку
    cat_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    heat["dow_name"] = pd.Categorical(heat["dow_name"], categories=cat_order, ordered=True)
    heat = heat.sort_values(["dow_name", "pickup_hour"])

    fig_heat = px.density_heatmap(
        heat,
        x="pickup_hour",
        y="dow_name",
        z="abs_error",
        histfunc="avg",
        labels={
            "pickup_hour": "Час суток",
            "dow_name": "День недели",
            "abs_error": "Средняя |ошибка|",
        },
        title="Средняя абсолютная ошибка по дню недели и часу",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ==========================
    #   РАСПРЕДЕЛЕНИЕ ОШИБОК
    # ==========================
    st.markdown("## Распределение ошибок")

    fig_hist = px.histogram(
        df_filt,
        x="error",
        nbins=50,
        labels={"error": "Ошибка (факт − прогноз)"},
        title="Распределение ошибок модели",
    )
    st.plotly_chart(fig_hist, use_container_width=True)


if __name__ == "__main__":
    main()
