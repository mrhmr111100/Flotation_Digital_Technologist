import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from utils import compute_metrics

RANDOM_STATE = 42

BEST_PARAMS = {
    'hidden_layer_sizes': (128, 64),
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 1e-2,
    'learning_rate_init': 5e-3,
    'batch_size': 64,
    'max_iter': 2000,
    'early_stopping': True,
    'validation_fraction': 0.15,
    'n_iter_no_change': 30,
    'random_state': RANDOM_STATE,
}


def train_and_evaluate_model(x_train, x_test, y_train, y_test):
    y_train_arr = np.asarray(y_train, dtype=float)
    y_mean = float(y_train_arr.mean())
    y_std = float(y_train_arr.std()) if y_train_arr.std() > 0 else 1.0

    scaler_y = StandardScaler()
    y_train_clipped = np.clip(y_train_arr, y_mean - 5 * y_std, y_mean + 5 * y_std).reshape(-1, 1)
    y_train_scaled = scaler_y.fit_transform(y_train_clipped).ravel()

    model = MLPRegressor(**BEST_PARAMS)
    model.fit(np.asarray(x_train, dtype=float), y_train_scaled)

    y_pred_train = scaler_y.inverse_transform(model.predict(np.asarray(x_train, dtype=float)).reshape(-1, 1)).ravel()
    y_pred_test = scaler_y.inverse_transform(model.predict(np.asarray(x_test, dtype=float)).reshape(-1, 1)).ravel()
    y_pred_train = np.clip(y_pred_train, y_mean - 5 * y_std, y_mean + 5 * y_std)
    y_pred_test = np.clip(y_pred_test, y_mean - 5 * y_std, y_mean + 5 * y_std)

    return model, {
        'Best Params': dict(BEST_PARAMS),
        **{f'Train {k}': v for k, v in compute_metrics(y_train_arr, y_pred_train).items()},
        **{f'Test {k}': v for k, v in compute_metrics(np.asarray(y_test, dtype=float), y_pred_test).items()},
    }
