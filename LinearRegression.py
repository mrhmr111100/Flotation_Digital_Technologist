import numpy as np
from sklearn.linear_model import Lasso, Ridge

from utils import compute_metrics

RANDOM_STATE = 42
LASSO_ALPHA = 0.5455594781168515
RIDGE_ALPHA = 0.37926901907322497


def train_and_evaluate_models(x_train, x_test, y_train, y_test):
    trained = {}
    results = {}

    lasso = Lasso(alpha=LASSO_ALPHA, max_iter=20000, random_state=RANDOM_STATE)
    lasso.fit(x_train, y_train)
    trained['Lasso'] = lasso
    results['Lasso'] = {
        'Best Params': {'alpha': float(LASSO_ALPHA)},
        **{f'Train {k}': v for k, v in compute_metrics(y_train, lasso.predict(x_train)).items()},
        **{f'Test {k}': v for k, v in compute_metrics(y_test, lasso.predict(x_test)).items()},
    }

    ridge = Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_STATE)
    ridge.fit(x_train, y_train)
    trained['Ridge'] = ridge
    results['Ridge'] = {
        'Best Params': {'alpha': float(RIDGE_ALPHA)},
        **{f'Train {k}': v for k, v in compute_metrics(y_train, ridge.predict(x_train)).items()},
        **{f'Test {k}': v for k, v in compute_metrics(y_test, ridge.predict(x_test)).items()},
    }

    return trained, results
