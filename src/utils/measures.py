import numpy as np
from sklearn.metrics import root_mean_squared_error


def rrmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    rmse = root_mean_squared_error(y_true, y_pred)

    mean_y_true = np.mean(y_true)

    return rmse / mean_y_true
