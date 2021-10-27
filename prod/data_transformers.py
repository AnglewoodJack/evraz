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

    def __init__(self, categorical_features: List[str], targets: List[str], alpha: float = 50.0):
        self.__is_fitted = False
        self.alpha = alpha
        self.mean_ = {}
        self.categorical_features = categorical_features
        self.mean_by_cat = {}
        self.encoded_prefix = "encoded_"
        self.targets = targets
        self.new_cat_features = None

    def smoothed_target_encoding(self, y: pd.Series, target_name: str) -> pd.Series:
        """
        Реализация регуляризованного target encoding.
        Чем меньше исходных данных, тем сильнее регуляризация.
        Параметр регуляризации регуляризует мин. кол-во необходимых данных.
        :param y: pd.Series с целевой переменной.
        :return: pd.Series с регуляризованной целевой переменной
        """
        n_rows = y.notnull().sum()
        return (y.mean() * n_rows + self.alpha * self.mean_[target_name]) / (n_rows + self.alpha)

    def fit(self, x: pd.DataFrame, y: Union[pd.DataFrame] = None):
        """
        На основе обучающей выборки запоминает средние значения целевой переменной в разрезе категорий.
        :param x: pd.DataFrame, обучающая выборка
        :param y: target
        :return:
        """
        x = pd.DataFrame(data=x, columns=self.categorical_features)
        # STE для каждой целевой переменной.
        for target in self.targets:
            # Присоединение целевой переменной к датафрейму.
            x[target] = y[target]
            # Среднее по всему target.
            self.mean_[target] = x[target].mean()
            # Значения по категориям каждого target-а.
            self.mean_by_cat[target] = {}
            # Среднее по target для каждой категории из соответствующего категориального признака.
            for col in self.categorical_features:
                # Значения по категориям.
                self.mean_by_cat[target].update( {
                    col: x.groupby(str(col))[target].apply(lambda row: self.smoothed_target_encoding(row, target)).fillna(self.mean_[target])
                })
        # Удаление целевой переменной из датафрейму.
        x.drop(self.targets, axis=1, inplace=True)
        self.new_cat_features = x.columns
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
        x_ = pd.DataFrame(data=x, columns=self.categorical_features)
        # Кодирование признаков.
        if self.__is_fitted:
            encoded_cols = []
            for target in self.targets:
                for col in self.categorical_features:
                    # Новые названия признаков.
                    new_col = self.encoded_prefix + target + '_' + col
                    # Новые признаки.
                    x_[new_col] = x_[col].map(self.mean_by_cat[target][col]).fillna(self.mean_[target])
                    encoded_cols.append(new_col)
            return x_[encoded_cols]
        else:
            # Ошибка, если fit не был выполнен.
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                f"Call 'fit' with appropriate arguments before using this transformer"
            )
