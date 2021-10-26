import pickle
import logging
import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from typing import List, Union, Dict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

from .data_transformers import SmoothedTargetEncoding

logger = logging.getLogger(__name__)


# TODO: доработать модель.
class PredictionModel:
    """
    Модель для прдсказания содержания углерода и температуры чугуна во время процесса продувки металла.
    :param numerical_features: list, список численных признаков из датафрейма
    :param ohe_categorical_features: list, список категориальных признаков для one-hot-encoding
    :param ste_categorical_features: list, список категориальных признаков для smoothed target encoding.
    :param lb_categorical_features: list, список категориальных признаков для label encoding.
    :param model_params: параметры модели.
    """

    def __init__(self,
                 numerical_features: List[str],
                 ohe_categorical_features: List[str],
                 ste_categorical_features: List[str],
                 # lb_categorical_features: List[str],
                 model_params: Dict[str, Union[str, int, float, list]]
                 ):
        self.num_features = numerical_features
        self.ohe_cat_features = ohe_categorical_features
        self.ste_cat_features = ste_categorical_features
        self._is_fitted = False
        # Трансформер вещественных признаков.
        self.numeric_transformer = Pipeline(
            steps=[
            ('num_imputer', SimpleImputer(strategy='median')), # заполнение пропусков.
            ('num_scaler', StandardScaler()) # нормализация.
            ])
        # Трансформеры категориальных признаков.
        self.ohe_transformer = Pipeline(
            steps=[
                ('ohe_imputer', SimpleImputer(strategy='constant')), # заполнение пропусков.
                ('ohe_encoder', OneHotEncoder())
            ])
        self.ste_transformer = Pipeline(
            steps=[
                ('ste_imputer', SimpleImputer(strategy='constant')), # заполнение пропусков.
                ('ste_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)), # энкодинг.
                # ('ste_encoder', SmoothedTargetEncoding(alpha=50)) # энкодинг.
            ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.numeric_transformer, self.num_features),
                ('ohe', self.ohe_transformer, self.ohe_cat_features),
                ('ste', self.ste_transformer, self.ste_cat_features)
            ])
        # Модель.
        self.model = LGBMRegressor(**model_params)
        # Пайплайн.
        self.pipeline = Pipeline(
            steps=[
                ('preprocessor', self.preprocessor),
                ('model', self.model)
            ])

    def fit(self, x: pd.DataFrame, y: Union[pd.Series, pd.DataFrame]):
        """
        Обучение модели.
        :param x: pd.DataFrame, датфрейм с признаками.
        :param y: pd.Series или pd.DataFrame, массив или датафрейм с целевыми переменными.
        """
        logger.info('Fit model ' + self.model.__module__)
        # TODO: Проверка работоспособности пайплайна в целом и SmoothedTargetEncoding в частности.
        # model__feature_name=[f'{i}' for i in range(70)],model__categorical_feature=['67','68','69']
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
        :return: Модель
        """
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model
