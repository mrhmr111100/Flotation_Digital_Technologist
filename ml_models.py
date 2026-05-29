from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class ModelError(Exception):
    """Ошибка при подготовке данных или обучении модели."""


@dataclass
class PredictionResult:
    model_name: str
    prediction: float
    metrics: dict
    prediction_time: float


def normalize_model_name(model_name: str) -> str:
    name = model_name.lower()

    if "forest" in name or "лес" in name:
        return "random_forest"

    if "linear" in name or "линей" in name or "regression" in name:
        return "linear_regression"

    raise ModelError(
        f"Модель «{model_name}» пока не подключена. "
        "На текущем этапе доступны Linear Regression и Random Forest."
    )

def create_model(model_key: str):
    if model_key == "random_forest":
        # адаптация Random Forest из репозитория mineral_enrichment
        # в исходном варианте модель использовалась как отдельный скрипт,
        # здесь она подключена к интерфейсу и обучается на выбранных пользователем столбцах
        return RandomForestRegressor(
            n_estimators=10,
            max_depth=10,
            random_state=42,
        )

    if model_key == "linear_regression":
        return LinearRegression()

    raise ModelError(f"Модель «{model_key}» не поддерживается.")


def readable_model_name(model_key: str) -> str:
    names = {
        "linear_regression": "Linear Regression",
        "random_forest": "Random Forest",
    }

    return names.get(model_key, model_key)


def clean_numeric_data(df, target_column, feature_columns):
    columns = [target_column] + feature_columns
    data = df[columns].copy()

    for column in columns:
        data[column] = (
            data[column]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .str.replace("\u00a0", "", regex=False)
            .str.strip()
        )
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna()

    if data.empty:
        raise ModelError("В выбранных столбцах нет числовых строк после очистки.")

    if len(data) < 3:
        raise ModelError("Для обучения модели нужно минимум 3 числовые строки.")

    return data


def prepare_training_data(df, target_column, feature_columns):
    data = clean_numeric_data(df, target_column, feature_columns)

    x = data[feature_columns].to_numpy(dtype=float)
    y = data[target_column].to_numpy(dtype=float)

    return x, y


def split_training_data(x, y):
    if len(y) < 5:
        return x, x, y, y, False

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    return x_train, x_test, y_train, y_test, True


def safe_mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = y_true != 0
    if not np.any(mask):
        return None

    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def calculate_metrics(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": safe_mape(y_true, y_pred),
    }


def train_and_predict(
    df,
    target_column,
    feature_columns,
    input_values,
    model_name,
):
    model_key = normalize_model_name(model_name)

    if len(input_values) != len(feature_columns):
        raise ModelError("Количество введённых параметров не совпадает с выбранными признаками.")

    x, y = prepare_training_data(df, target_column, feature_columns)
    x_train, x_test, y_train, y_test, has_test = split_training_data(x, y)

    model = create_model(model_key)
    model.fit(x_train, y_train)

    x_input = np.array(input_values, dtype=float).reshape(1, -1)

    started = perf_counter()
    prediction = float(model.predict(x_input)[0])
    prediction_time = perf_counter() - started

    metrics = {}
    if has_test:
        y_pred = model.predict(x_test)
        metrics = calculate_metrics(y_test, y_pred)

    return PredictionResult(
        model_name=readable_model_name(model_key),
        prediction=prediction,
        metrics=metrics,
        prediction_time=prediction_time,
    )