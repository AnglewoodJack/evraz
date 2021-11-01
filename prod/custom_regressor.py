import numpy as np
import pandas as pd
from typing import Union
from catboost import CatBoostRegressor
from sklearn.feature_selection import SelectFromModel
from prod.settings import CAT_PREFIX, TARGET_LOG


class CustomCatBoostRegressor(CatBoostRegressor):
    """
    Класс, наследуемый функционал от CatBoostRegressor - для возможности учета изменяемых в процессе
    трансформации данных (encoding/feature engineering) имен признаковых переменных (feature names).
    """

    def fit(self, x: Union[np.array, pd.Series, pd.DataFrame],
            y: Union[np.array, pd.Series, pd.DataFrame] = None, **kwargs):

        return super().fit(
            x,
            y = np.log(y) if TARGET_LOG else y,
            cat_features=x.filter(regex=f"^{CAT_PREFIX}").columns.to_list(),
            **kwargs
        )


class CustomFeatureSelection(SelectFromModel):
    """
    Класс, наследуемый функционал от SelectFromModel - для отбора изменяемых в процессе
    трансформации данных (encoding/feature engineering) имен признаковых переменных (feature names).
    """
    def transform(self, x):
        # Индексы для необходимых признаков.
        important_features_indices = list(self.get_support(indices=False))
        # Отбор признаков
        _x = x.iloc[:, important_features_indices].copy()
        return _x
