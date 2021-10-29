from catboost import CatBoostRegressor
from sklearn.feature_selection import SelectFromModel

from prod.settings import CAT_PREFIX


class CustomCatBoostRegressor(CatBoostRegressor):

    def fit(self, x, y=None, **kwargs):
        print(x.filter(regex=f"^{CAT_PREFIX}").columns.to_list())

        return super().fit(
            x,
            y=y,
            cat_features=x.filter(regex=f"^{CAT_PREFIX}").columns.to_list(),
            **kwargs
        )


class CustomFeatureSelection(SelectFromModel):

    def transform(self, x):

        # Get indices of important features
        important_features_indices = list(self.get_support(indices=False))

        # Select important features
        _x = x.iloc[:, important_features_indices].copy()

        return _x
