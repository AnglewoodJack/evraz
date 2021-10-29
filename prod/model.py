import pickle
import logging
import numpy as np
import pandas as pd
from typing import Union, Dict
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError
from prod.custom_regressor import CustomCatBoostRegressor, CustomFeatureSelection

logger = logging.getLogger(__name__)


class PredictionModel:
    """
    Модель для предсказания содержания углерода и температуры чугуна во время процесса продувки металла.
    :param numerical_features: list, список численных признаков из датафрейма
    :param ohe_categorical_features: list, список категориальных признаков для one-hot-encoding
    :param ste_categorical_features: list, список категориальных признаков для smoothed target encoding.
    :param model_params: параметры модели.
    """
    __is_fitted: bool

    def __init__(self, mapper: object, model_params: Dict[str, Union[str, int, float, list]]):
        self._is_fitted = False
        self.pipeline = Pipeline(steps=[
            ("preprocess", mapper),
            ("feature_selection", CustomFeatureSelection(CustomCatBoostRegressor(**model_params))),
            ("estimator", CustomCatBoostRegressor(**model_params))
             ])

    def fit(self, x: pd.DataFrame, y: Union[pd.Series, pd.DataFrame]):
        """
        Обучение модели.
        :param x: pd.DataFrame, датфрейм с признаками.
        :param y: pd.Series или pd.DataFrame, массив или датафрейм с целевыми переменными.
        """
        logger.info('Fit model ' + self.pipeline['estimator'].__module__)
        self.pipeline.fit(x, y)
        logger.info('Model fit completed successfully.')
        self.__is_fitted = True

    def predict(self, x: pd.DataFrame) -> np.array:
        """
        Предсказание модели.
        :param x: pd.DataFrame, признаки.
        :return: np.array, предсказания.
        """
        if self.__is_fitted:
            return self.pipeline.predict(x)
        else:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet! "
                f"Call 'fit' with appropriate arguments before predict"
            )

    def save(self, path: str):
        """
        Сериализация модели в pickle.
        :param path: str, путь к файлу.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        """
        Загрузка модели из pickle.
        :param path: str, путь к файлу.
        :return: Модель.
        """
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model
