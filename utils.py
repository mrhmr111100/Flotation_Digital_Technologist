"""Общие утилиты для ML-моделей — метрики, helpers."""

import numpy as np


def smape(y_true, y_pred):
    """Symmetric MAPE."""
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(yt) & np.isfinite(yp)
    if not mask.any():
        return float('nan')
    denom = np.abs(yt[mask]) + np.abs(yp[mask])
    nz = denom > 0
    if not nz.any():
        return 0.0
    return float(np.mean(2.0 * np.abs(yt[mask][nz] - yp[mask][nz]) / denom[nz]))


def safe_mape(y_true, y_pred):
    """MAPE с защитой от деления на ноль."""
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(yt) & np.isfinite(yp) & (yt != 0)
    if not mask.any():
        return None
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])))


def compute_metrics(y_true, y_pred):
    """Вычислить все метрики за один вызов. Возвращает dict."""
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    mse = mean_squared_error(y_true, y_pred)
    return {
        'R2': r2_score(y_true, y_pred),
        'MSE': mse,
        'RMSE': mse ** 0.5,
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE': safe_mape(y_true, y_pred),
        'SMAPE': smape(y_true, y_pred),
    }
