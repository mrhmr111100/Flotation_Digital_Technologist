from importlib import import_module
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Dict, Callable

import ml_models

MODULE_REGISTRY: Dict[str, ModuleType] = {}


def _load_module_from_path(path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from path: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def register_module(path_or_name: str, alias: str = None):
    """Register a module by dotted name or file path.

    alias — ключ, под которым модель будет доступна (по умолчанию имя модуля).
    """
    if Path(path_or_name).exists():
        mod = _load_module_from_path(path_or_name)
    else:
        mod = import_module(path_or_name)

    name = alias or getattr(mod, "MODEL_NAME", None) or mod.__name__.split(".")[-1]
    MODULE_REGISTRY[name] = mod
    return name


def register_common_modules():
    """Попытаться зарегистрировать модули по стандартным именам, если они доступны в PYTHONPATH
    или как соседние файлы в рабочей директории проекта (mineral_enrichment и т.п.).
    """
    candidates = [
        "RandomForest",
        "GradientBoosting",
        "PLS",
        "GPR",
        "MLP",
        "LinearRegression",
    ]
    registered = []
    for c in candidates:
        try:
            register_module(c)
            registered.append(c)
            continue
        except Exception:
            pass

        # Попробуем файл в той же директории, что и model_backend
        try:
            p = Path(__file__).parent / (c + ".py")
            if p.exists():
                register_module(str(p), alias=c)
                registered.append(c)
                continue
        except Exception:
            pass

        # Попытка подгрузить файл из соседнего репозитория (старое поведение)
        try:
            p2 = Path(__file__).parent.parent / c / (c + ".py")
            if p2.exists():
                register_module(str(p2), alias=c)
                registered.append(c)
        except Exception:
            continue
    return registered


def available_models():
    return list(MODULE_REGISTRY.keys())


def _wrap_train_and_predict_from_evaluate(module: ModuleType) -> Callable:
    """Возвращает функцию train_and_predict(df, target_column, feature_columns, input_values)
    для модулей, у которых есть `train_and_evaluate_model(x_train, x_test, y_train, y_test)`.
    """
    def fn(df, target_column, feature_columns, input_values):
        # Подготовка данных: используем функции из ml_models для очистки и разбиения
        data = ml_models.clean_numeric_data(df, target_column, feature_columns)
        x = data[feature_columns].to_numpy(dtype=float)
        y = data[target_column].to_numpy(dtype=float)
        x_train, x_test, y_train, y_test, has_test = ml_models.split_training_data(x, y)

        # Ищем функцию train_and_evaluate_model
        if hasattr(module, "train_and_evaluate_model"):
            model, metrics = module.train_and_evaluate_model(x_train, x_test, y_train, y_test)
        else:
            raise ml_models.ModelError("Модуль не содержит поддерживаемой точки входа train_and_evaluate_model")

        # Предсказание на входных значениях — пытаемся вызвать predict у модели
        import numpy as _np
        x_input = _np.array(input_values, dtype=float).reshape(1, -1)
        try:
            pred = float(model.predict(x_input)[0])
        except Exception:
            pred = None

        return ml_models.PredictionResult(
            model_name=getattr(module, "__name__", "plugin"),
            prediction=pred,
            metrics=(metrics if isinstance(metrics, dict) else {}),
            prediction_time=0.0,
        )

    return fn


def train_and_predict(df, target_column, feature_columns, input_values, model_name):
    """Унифицированная точка входа — аналогична `ml_models.train_and_predict`.
    """
    # Попробуем гибко найти зарегистрированную модель по имени, учитывая пробелы, регистр и подчеркивания
    def _normalize(s: str) -> str:
        import re
        return re.sub(r'[^0-9a-z]', '', str(s).lower())

    if model_name not in MODULE_REGISTRY:
        norm = _normalize(model_name)
        found = None
        for key in MODULE_REGISTRY.keys():
            if _normalize(key) == norm or norm in _normalize(key) or _normalize(key) in norm:
                found = key
                break
        if found is None:
            raise ml_models.ModelError(f"Модель {model_name} не зарегистрирована в model_backend")
        model_name = found

    mod = MODULE_REGISTRY[model_name]

    # Если модуль уже реализует train_and_predict — вызываем напрямую
    if hasattr(mod, "train_and_predict"):
        return mod.train_and_predict(df, target_column, feature_columns, input_values)

    # Иначе, если есть train_and_evaluate_model, оборачиваем
    if hasattr(mod, "train_and_evaluate_model"):
        fn = _wrap_train_and_predict_from_evaluate(mod)
        return fn(df, target_column, feature_columns, input_values)

    raise ml_models.ModelError("Модуль не содержит поддерживаемых точек входа (train_and_predict или train_and_evaluate_model)")
