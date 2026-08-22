import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from utils import compute_metrics

RANDOM_STATE = 42
MAX_COMPONENTS_CAP = 30
CV_SPLITS = 5


def _cv_select_n_components(X, y, max_components):
    n = X.shape[0]
    kf = KFold(n_splits=min(CV_SPLITS, max(2, n // 20)), shuffle=True, random_state=RANDOM_STATE)
    best_comp = 1
    best_mse = np.inf
    for nc in range(1, max_components + 1):
        mses = []
        for tr_idx, va_idx in kf.split(X):
            pls = PLSRegression(n_components=nc)
            pls.fit(X[tr_idx], y[tr_idx])
            pred = pls.predict(X[va_idx])
            mses.append(mean_squared_error(y[va_idx], pred))
        avg = float(np.mean(mses))
        if avg < best_mse:
            best_mse = avg
            best_comp = nc
    return best_comp, best_mse


def train_and_evaluate_model(x_train, x_test, y_train, y_test):
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(np.asarray(x_train, dtype=float))
    X_test_scaled = scaler_X.transform(np.asarray(x_test, dtype=float))

    scaler_y = StandardScaler()
    y_arr = np.asarray(y_train, dtype=float).reshape(-1, 1)
    y_train_scaled = scaler_y.fit_transform(y_arr).ravel()

    max_comp = min(X_train_scaled.shape[1], X_train_scaled.shape[0] - 2, MAX_COMPONENTS_CAP)
    max_comp = max(1, max_comp)

    best_comp, cv_mse = _cv_select_n_components(X_train_scaled, y_train_scaled, max_comp)

    pls_final = PLSRegression(n_components=best_comp)
    pls_final.fit(X_train_scaled, y_train_scaled)

    y_pred_train = scaler_y.inverse_transform(pls_final.predict(X_train_scaled).reshape(-1, 1)).ravel()
    y_pred_test = scaler_y.inverse_transform(pls_final.predict(X_test_scaled).reshape(-1, 1)).ravel()

    y_train_arr = np.asarray(y_train, dtype=float)
    y_test_arr = np.asarray(y_test, dtype=float)

    return pls_final, {
        'Best Params': {'n_components': int(best_comp), 'max_components_considered': int(max_comp)},
        'CV best score (MSE)': float(cv_mse),
        **{f'Train {k}': v for k, v in compute_metrics(y_train_arr, y_pred_train).items()},
        **{f'Test {k}': v for k, v in compute_metrics(y_test_arr, y_pred_test).items()},
    }
