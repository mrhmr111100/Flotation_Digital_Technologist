import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from utils import compute_metrics

RANDOM_STATE = 42

BEST_PARAMS = {
    'n_restarts_optimizer': 3,
    'alpha': 1e-2,
    'normalize_y': True,
    'random_state': RANDOM_STATE,
}

KERNEL = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-1)


def build_model():
    return GaussianProcessRegressor(kernel=KERNEL, **BEST_PARAMS)


def train_and_evaluate_model(x_train, x_test, y_train, y_test):
    model = build_model()
    model.fit(x_train, y_train)

    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    return model, {
        'Best Params': {
            **BEST_PARAMS,
            'kernel': 'ConstantKernel(1.0)*RBF(1.0)+WhiteKernel(1e-1)',
        },
        **{f'Train {k}': v for k, v in compute_metrics(y_train, y_pred_train).items()},
        **{f'Test {k}': v for k, v in compute_metrics(y_test, y_pred_test).items()},
    }
