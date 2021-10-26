import numpy as np
import pandas as pd
from typing import Union
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator, TransformerMixin


class SmoothedTargetEncoding(BaseEstimator, TransformerMixin):
    """
    Регуляризованный target encoding.
    :param categorical_features: список из столбцов с категориальными признаками для энкодинга.
    :param alpha: параметр регуляризации.
    """

    def __init__(self, alpha: float = 50.0):
        self.__is_fitted = False
        self.alpha = alpha
        self.mean_ = None
        self.categorical_features = None
        self.mean_by_cat = {}
        self.encoded_prefix = "encoded_"
        self.target = 'target'

    def smoothed_target_encoding(self, y: pd.Series) -> pd.Series:
        """
        Реализация регуляризованного target encoding.
        Чем меньше исходных данных, тем сильнее регуляризация.
        Параметр регуляризации регуляризует мин. кол-во необходимых данных.
        :param y: pd.Series с целевой переменной.
        :return: pd.Series с регуляризованной целевой переменной
        """
        n_rows = y.notnull().sum()
        return (y.mean() * n_rows + self.alpha * self.mean_) / (n_rows + self.alpha)

    def fit(self, x: pd.DataFrame, y: Union[np.array, pd.Series] = None):
        """
        На основе обучающей выборки запоминает средние значения целевой переменной в разрезе категорий.
        :param x: pd.DataFrame, обучающая выборка
        :param y: target
        :return:
        """
        # Названия всех категориальных признаков.
        self.categorical_features = x.columns
        # Присоединение целевой переменной к датафрейму.
        x[self.target] = y
        # Среднее по всему target.
        self.mean_ = x[self.target].mean()
        # Среднее по target для каждой категории из соответствующего категориального признака.
        for col in self.categorical_features:
            # Значения по категориям.
            self.mean_by_cat[col] = (
                x.groupby(col)[self.target].apply(lambda n: self.smoothed_target_encoding(n)).fillna(self.mean_)
            )
        # Удалени целевой переменной из датафрейму.
        x.drop(self.target, axis=1, inplace=True)
        # Флаг успешного завершениия.
        self.__is_fitted = True
        return self

    def transform(self, x: pd.DataFrame):
        """
        Применение регуляризованного target encoding.
        :param x: pd.DataFrame, обучающая выборка.
        :return: pd.DataFrame, закодированные данные
        """
        # Создание копии датафрейма.
        x_cp = x.copy()
        # Кодирование прзнаков.
        if self.__is_fitted:
            encoded_cols = []
            for col in self.categorical_features:
                # Новые названия признаков.
                new_col = self.encoded_prefix + col
                # Новые признаки.
                x_cp[new_col] = x_cp[col].map(self.mean_by_cat[col]).fillna(self.mean_)
                encoded_cols.append(new_col)
            return x_cp[encoded_cols]
        else:
            # Ошибка, если fit не был выполнен.
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                f"Call 'fit' with appropriate arguments before using this transformer"
            )
