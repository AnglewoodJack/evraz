import numpy as np
import pandas as pd
from typing import Union, List
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator, TransformerMixin


class SmoothedTargetEncoding(BaseEstimator, TransformerMixin):
    """
    Регуляризованный target encoding.
    :param alpha: параметр регуляризации.
    """

    def __init__(self, target: str, alpha: float = 50.0):
        self.__is_fitted = False
        self.alpha = alpha
        self.target = target
        self.mean_ = None
        self.feature = None
        self.mean_by_cat = {}
        self.encoded_prefix = "encoded_"

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

    def fit(self, x: pd.DataFrame, y: Union[pd.DataFrame] = None):
        """
        На основе обучающей выборки запоминает средние значения целевой переменной в разрезе категорий.
        :param x: pd.DataFrame, обучающая выборка
        :param y: target
        :return:
        """
        print(x)
        x = pd.DataFrame(data=x, columns=['value'])
        print(x)
        # STE для каждой целевой переменной.
        # Присоединение целевой переменной к датафрейму.
        x[self.target] = y[self.target]
        print(x)
        # Среднее по всему target.
        self.mean_ = x[self.target].mean()
        print(self.mean_)
        # Среднее по target для каждой категории из соответствующего категориального признака.
        # Значения по категориям.
        self.mean_by_cat = (
            x.groupby('value')[self.target]
             .apply(lambda row: self.smoothed_target_encoding(row))
             .fillna(self.mean_)
        )
        print(self.mean_by_cat)
        # Удаление целевой переменной из датафрейму.
        x.drop(self.target, axis=1, inplace=True)
        print(x)
        # Флаг успешного завершения.
        self.__is_fitted = True
        return self

    def transform(self, x: pd.DataFrame):
        """
        Применение регуляризованного target encoding.
        :param x: pd.DataFrame, обучающая выборка.
        :return: pd.DataFrame, закодированные данные
        """
        # Создание копии датафрейма.
        x_ = x.copy()
        # Кодирование признаков.
        if self.__is_fitted:
            # Новые названия признаков.
            new_col = self.encoded_prefix + self.feature
            # Новые признаки.
            x_[new_col] = x_.map(self.mean_by_cat[self.target]).fillna(self.mean_)
            return x_[[new_col]]
        else:
            # Ошибка, если fit не был выполнен.
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                f"Call 'fit' with appropriate arguments before using this transformer"
            )
