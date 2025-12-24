import os
import numpy as np
import pandas as pd

from sqlalchemy import create_engine
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

import optuna
from optuna.samplers import TPESampler

from catboost import CatBoostRegressor, Pool

PG_URL_SQLA = "postgresql+psycopg2://postgres:1234@localhost:5433/nyc_taxi"


def load_data():
    engine = create_engine(PG_URL_SQLA)

    df = pd.read_sql("SELECT * FROM ml_trips_hourly", engine)
    df["pickup_ts"] = pd.to_datetime(df["pickup_ts"])
    df = df.sort_values("pickup_ts").reset_index(drop=True)

    target_col = "trips_count"

    feature_cols = [
        "pickup_hour",
        "pickup_dow",
        "is_weekend",
        "trips_lag_1d",
        "trips_lag_7d",
        "trips_ma_7d",
        "hour_sin",
        "hour_cos",
        "PU_borough",
        "PU_zone",
    ]

    X = df[feature_cols]
    y = df[target_col]

    cat_features = ["PU_borough", "PU_zone"]

    return df, X, y, feature_cols, target_col, cat_features


def create_timeseries_splits(X, n_splits=3):
    """
    TimeSeriesSplit: делим по времени, как приличные люди.
    """
    tss = TimeSeriesSplit(n_splits=n_splits)
    splits = list(tss.split(X))
    return splits


def objective(trial, X, y, cat_features, splits):
    """
    Целевая функция для Optuna: подбираем гиперпараметры CatBoost.
    """

    # Гиперпараметры, которые будем тюнить
    params = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": 42,
        "thread_count": -1,
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "border_count": trial.suggest_int("border_count", 64, 255),
        "od_type": "Iter",
        "od_wait": 50,
        "iterations": 800,  # на тюнинге лучше поменьше, потом можно увеличить
        "verbose": False,
    }

    rmses = []

    for fold_id, (train_idx, valid_idx) in enumerate(splits, start=1):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        valid_pool = Pool(X_valid, y_valid, cat_features=cat_features)

        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

        y_pred = model.predict(valid_pool)
        rmse = mean_squared_error(y_valid, y_pred, squared=False)
        rmses.append(rmse)

    mean_rmse = float(np.mean(rmses))
    trial.set_user_attr("rmse_per_fold", rmses)

    return mean_rmse



def main():
    print(">>> START train_trips_catboost_optuna.py")

    # --- Проверяем, есть ли уже обученная модель ---
    if os.path.exists("models/catboost_trips_hourly_optuna.cbm"):
        print("Модель уже существует → загружаем её и пропускаем обучение.")
        model = CatBoostRegressor()
        model.load_model("models/catboost_trips_hourly_optuna.cbm")
        print("Модель успешно загружена.")
        return  # выходим — тюнинг и обучение не запускаются

    df, X, y, feature_cols, target_col, cat_features = load_data()
    splits = create_timeseries_splits(X, n_splits=3)

    # ---- Optuna study ----
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        study_name="catboost_trips_hourly",
        direction="minimize",
        sampler=sampler,
    )

    def wrapped_objective(trial):
        return objective(trial, X, y, cat_features, splits)

    # Кол-во прогонов можно увеличить, но сначала попробуй 20–30
    study.optimize(wrapped_objective, n_trials=25, show_progress_bar=True)

    print("\n==== Optuna best trial ====")
    best_trial = study.best_trial
    print("Best RMSE:", best_trial.value)
    print("Best params:")
    for k, v in best_trial.params.items():
        print(f"  {k}: {v}")
    print("RMSE per fold:", best_trial.user_attrs.get("rmse_per_fold"))

    # ---- Обучаем финальную модель на ВСЁМ датасете с лучшими параметрами ----
    best_params = best_trial.params.copy()
    best_params.update(
        {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "random_seed": 42,
            "thread_count": -1,
            "iterations": 2000,  # здесь можно увеличить
            "od_type": "Iter",
            "od_wait": 100,
            "verbose": 200,
        }
    )

    full_pool = Pool(X, y, cat_features=cat_features)
    final_model = CatBoostRegressor(**best_params)
    final_model.fit(full_pool)

    os.makedirs("models", exist_ok=True)
    model_path = "models/catboost_trips_hourly_optuna.cbm"
    final_model.save_model(model_path)
    print(f"\nФинальная модель сохранена в {model_path}")

    # ---- Важность признаков ----
    fi = final_model.get_feature_importance(full_pool, type="FeatureImportance")
    fi_df = pd.DataFrame({"feature": feature_cols, "importance": fi}).sort_values(
        "importance", ascending=False
    )
    print("\nFeature importance:")
    print(fi_df)


if __name__ == "__main__":
    main()
