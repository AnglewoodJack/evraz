import numpy as np
import pandas as pd
from typing import Dict
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error


def evraz_metric(y_true: pd.DataFrame, y_pred: np.array):
    """
    Метрика оценки качества модели, предложенная организаторами EVRAZ.
    :param answers: pd.DataFrame, датасет с реальными значениями целевых переменных.
    :param user_csv: pd.DataFrame, датасет с предсказанными значениями целевых переменных.
    :return:
    """
    predictions = pd.DataFrame(data=y_pred, columns=['C', 'TST'])
    # Содержание углерода в металле.
    delta_c = np.abs(np.array(y_true['C']) - np.array(predictions['C']))
    hit_rate_c = np.int64(delta_c < 0.02)
    # Температура металла.
    delta_t = np.abs(np.array(y_true['TST']) - np.array(predictions['TST']))
    hit_rate_t = np.int64(delta_t < 20)
    N = np.size(y_true['C'])
    return np.sum(hit_rate_c + hit_rate_t) / 2 / N


def median_absolute_percentage_error(y_true: np.array, y_pred: np.array) -> float:
    """
    MAPE метрика.
    :param y_true: np.array, реальные значения целевой переменной.
    :param y_pred: np.array, предсказанные значения целевой переменной.
    :return: score модели.
    """
    return np.median(np.abs(y_pred-y_true)/y_true)


def metrics_stat(y_true: np.array, y_pred: np.array) -> Dict[str, float]:
    """
    Вывод основных метрик.
    :param y_true: np.array, реальные значения целевой переменной.
    :param y_pred: np.array, предсказанные значения целевой переменной.
    :return: dict, словарь с названиями метрик и значениями
    """
    mape = mean_absolute_percentage_error(y_true, y_pred)
    mdape = median_absolute_percentage_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return {'mape': mape, 'mdape': mdape, 'rmse': rmse, 'r2': r2}
