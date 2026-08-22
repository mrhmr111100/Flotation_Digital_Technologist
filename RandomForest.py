































































import numpy as np
from sklearn.ensemble import RandomForestRegressor

from utils import compute_metrics

RANDOM_STATE = 42

BEST_PARAMS = {
    'n_estimators': 400,
    'max_depth': None,
    'min_samples_leaf': 1,
    'max_features': 0.5,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
}


def build_model():
    return RandomForestRegressor(**BEST_PARAMS)


def train_and_evaluate_model(x_train, x_test, y_train, y_test):
    model = build_model()
    model.fit(x_train, y_train)

    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    return model, {
        'Best Params': dict(BEST_PARAMS),
        **{f'Train {k}': v for k, v in compute_metrics(y_train, y_pred_train).items()},
        **{f'Test {k}': v for k, v in compute_metrics(y_test, y_pred_test).items()},
    }
