import numpy as np
import pandas as pd
from typing import Union
from category_encoders import TargetEncoder


class CustomTargetEncoder(TargetEncoder):
    """
    Регуляризованный энкодинг на основе целевой переменной - smoothed target encoding.
    Базовый класс расширен для возможности использования в пайплайне задачи многоцелевой
    регрессии (multi-target regression).
    :param target: целевая переменная, по которой осуществляется энкодинг.
    """

    def __init__(self, target: str, **kwargs):
        super().__init__()
        self.target = target

    def fit(self, x: Union[np.array, pd.Series, pd.DataFrame],
            y: Union[np.array, pd.Series, pd.DataFrame] = None, **kwargs):

        return super().fit(
            x,
            y=y[self.target],
            **kwargs
        )

    def transform(self, x: Union[np.array, pd.Series, pd.DataFrame],
                  y: Union[np.array, pd.Series, pd.DataFrame] = None, **kwargs):

        return super().transform(
            x,
            **kwargs
        )
