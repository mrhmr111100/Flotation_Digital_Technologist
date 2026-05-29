from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class ModelError(Exception):
    """Ошибка подготовки данных или обучения модели."""


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

    if "boost" in name or "буст" in name:
        return "gradient_boosting"

    if "ансамб" in name or "ensemble" in name:
        return "ensemble"

    return "linear_regression"


def make_model(model_key: str):
    if model_key == "random_forest":
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
        )

    if model_key == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        )

    if model_key == "linear_regression":
        return LinearRegression()

    raise ModelError(f"Модель «{model_key}» не поддерживается.")


def clean_training_data(df, target_column, feature_columns):
    columns = [target_column] + feature_columns
    data = df[columns].copy()

    for column in columns:
        data[column] = (
            data[column]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .str.replace("\u00a0", "", regex=False)
        )
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna()

    if data.empty:
        raise ModelError("В выбранных столбцах нет числовых строк после очистки.")

    if len(data) < 3:
        raise ModelError("Для обучения модели нужно минимум 3 числовые строки.")

    return data


def prepare_xy(df, target_column, feature_columns):
    data = clean_training_data(df, target_column, feature_columns)

    x = data[feature_columns].to_numpy(dtype=float)
    y = data[target_column].to_numpy(dtype=float)

    return x, y


def safe_mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = y_true != 0
    if not np.any(mask):
        return None

    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def calculate_metrics(y_true, y_pred):
    metrics = {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": safe_mape(y_true, y_pred),
    }

    return metrics


def split_data(x, y):
    if len(y) < 5:
        return x, x, y, y, False

    return (*train_test_split(x, y, test_size=0.2, random_state=42), True)


def train_single_model(model_key, x, y, x_input):
    x_train, x_test, y_train, y_test, has_test = split_data(x, y)

    model = make_model(model_key)
    model.fit(x_train, y_train)

    started = perf_counter()
    prediction = float(model.predict(x_input)[0])
    prediction_time = perf_counter() - started

    metrics = {}
    if has_test:
        y_pred = model.predict(x_test)
        metrics = calculate_metrics(y_test, y_pred)

    return prediction, metrics, prediction_time


def train_ensemble(x, y, x_input):
    model_keys = [
        "linear_regression",
        "random_forest",
        "gradient_boosting",
    ]

    predictions = []
    times = []
    metric_rows = []

    for model_key in model_keys:
        prediction, metrics, prediction_time = train_single_model(model_key, x, y, x_input)
        predictions.append(prediction)
        times.append(prediction_time)

        if metrics:
            metric_rows.append(metrics)

    metrics = average_metrics(metric_rows)

    return float(np.mean(predictions)), metrics, float(sum(times))


def average_metrics(metric_rows):
    if not metric_rows:
        return {}

    result = {}

    for key in ["R2", "RMSE", "MAE", "MAPE"]:
        values = [
            row[key]
            for row in metric_rows
            if key in row and row[key] is not None
        ]

        if values:
            result[key] = float(np.mean(values))

    return result


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

    x, y = prepare_xy(df, target_column, feature_columns)
    x_input = np.array(input_values, dtype=float).reshape(1, -1)

    if model_key == "ensemble":
        prediction, metrics, prediction_time = train_ensemble(x, y, x_input)
        title = "Ансамбль моделей"
    else:
        prediction, metrics, prediction_time = train_single_model(model_key, x, y, x_input)
        title = readable_model_name(model_key)

    return PredictionResult(
        model_name=title,
        prediction=prediction,
        metrics=metrics,
        prediction_time=prediction_time,
    )


def readable_model_name(model_key):
    names = {
        "linear_regression": "Linear Regression",
        "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting",
    }

    return names.get(model_key, model_key)