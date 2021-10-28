from catboost import CatBoostRegressor


class CustomCatBoostClassifier(CatBoostRegressor):

    def fit(self, X, y=None, **fit_params):
        print(X.filter(regex=f"{categorical_suffix}$").columns.to_list())

        return super().fit(
            X,
            y=y,
            cat_features=X.filter(regex=f"{categorical_suffix}$").columns,
            **fit_params
        )


