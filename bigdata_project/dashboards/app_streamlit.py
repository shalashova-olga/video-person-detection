import os
import pandas as pd
from sqlalchemy import create_engine
import streamlit as st
import plotly.express as px


# ---------- НАСТРОЙКИ ПОДКЛЮЧЕНИЯ К PG ----------
PG_USER = "postgres"
PG_PASSWORD = "1234"         
PG_HOST = "localhost"
PG_PORT = 5433                      
PG_DB = "nyc_taxi"                

CONN_STR = f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"


# ---------- ЗАГРУЗКА ДАННЫХ С КЭШЕМ ----------
@st.cache_data(show_spinner=True)
def load_data():
    engine = create_engine(CONN_STR)
    query = '''
    SELECT
        pickup_date,
        pickup_hour,
        pickup_dow,
        "PU_borough" AS "PU_borough",
        "PU_zone"    AS "PU_zone",
        is_weekend,
        day_segment,
        trips_count,
        avg_duration_min,
        avg_speed_mph,
        avg_total_amount,
        sum_revenue,
        avg_trip_distance,
        airport_pickups_count,
        airport_dropoffs_count
    FROM public.fact_trips_agg
'''
    df = pd.read_sql(query, engine)
    return df


# ---------- UI ----------
st.set_page_config(
    page_title="NYC Taxi Analytics",
    layout="wide",
)

st.title("🚕 NYC Taxi Analytics — Streamlit дашборд")
st.markdown("Агрегаты по поездкам такси из витрины `fact_trips_agg` (PostgreSQL).")

df = load_data()
df["pickup_date"] = pd.to_datetime(df["pickup_date"])

# Боковая панель с фильтрами
st.sidebar.header("Фильтры")

# Даты
min_date = df["pickup_date"].min()
max_date = df["pickup_date"].max()
date_range = st.sidebar.date_input(
    "Диапазон дат",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

# Borough
boroughs = sorted(df["PU_borough"].dropna().unique())
selected_boroughs = st.sidebar.multiselect(
    "Borough посадки",
    options=boroughs,
    default=boroughs,
)

# Будни/выходные
weekend_map = {0: "Будни", 1: "Выходные"}
weekend_choice = st.sidebar.multiselect(
    "Тип дня",
    options=[0, 1],
    format_func=lambda x: weekend_map[x],
    default=[0, 1],
)

# Применяем фильтры
filtered = df.copy()
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
    filtered = filtered[
    (filtered["pickup_date"] >= pd.Timestamp(start)) &
    (filtered["pickup_date"] <= pd.Timestamp(end))
]

if selected_boroughs:
    filtered = filtered[filtered["PU_borough"].isin(selected_boroughs)]

if weekend_choice:
    filtered = filtered[filtered["is_weekend"].isin(weekend_choice)]


# ---------- KPI ----------
st.subheader("Основные показатели")

total_trips = int(filtered["trips_count"].sum())
total_revenue = float(filtered["sum_revenue"].sum())
avg_check = float(filtered["avg_total_amount"].mean())

col1, col2, col3 = st.columns(3)
col1.metric("Всего поездок", f"{total_trips:,}".replace(",", " "))
col2.metric("Суммарная выручка, $", f"{total_revenue:,.0f}".replace(",", " "))
col3.metric("Средний чек, $", f"{avg_check:,.2f}")

st.markdown("---")

# ---------- Графики ----------
col_left, col_right = st.columns(2)

# 1) Кол-во поездок по часам
with col_left:
    st.markdown("### Количество поездок по часам суток")
    hourly = (
        filtered.groupby("pickup_hour")["trips_count"]
        .sum()
        .reset_index()
        .sort_values("pickup_hour")
    )
    fig_hourly = px.line(
        hourly,
        x="pickup_hour",
        y="trips_count",
        markers=True,
        labels={"pickup_hour": "Час суток", "trips_count": "Количество поездок"},
    )
    st.plotly_chart(fig_hourly, use_container_width=True)

# 2) Выручка по borough
with col_right:
    st.markdown("### Суммарная выручка по borough")
    rev_borough = (
        filtered.groupby("PU_borough")["sum_revenue"]
        .sum()
        .reset_index()
        .sort_values("sum_revenue", ascending=False)
    )
    fig_rev_borough = px.bar(
        rev_borough,
        x="PU_borough",
        y="sum_revenue",
        labels={"PU_borough": "Borough", "sum_revenue": "Выручка, $"},
    )
    st.plotly_chart(fig_rev_borough, use_container_width=True)

st.markdown("---")

# 3) ТОП-зоны по выручке
st.markdown("### Топ-10 зон по выручке")
top_zones = (
    filtered.groupby("PU_zone")["sum_revenue"]
    .sum()
    .reset_index()
    .sort_values("sum_revenue", ascending=False)
    .head(10)
)
fig_top_zones = px.bar(
    top_zones,
    x="sum_revenue",
    y="PU_zone",
    orientation="h",
    labels={"PU_zone": "Зона посадки", "sum_revenue": "Выручка, $"},
)
st.plotly_chart(fig_top_zones, use_container_width=True)

st.markdown("Данные берутся из PostgreSQL → витрина `public.fact_trips_agg`.")
