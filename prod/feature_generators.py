import pandas as pd
from sklearn_pandas import DataFrameMapper, gen_features
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder

from data_transformers import SmoothedTargetEncoding
from settings import CATEGORICAL_OHE_FEATURES, CATEGORICAL_STE_FEATURES, NUM_FEATURES, TARGET, CAT_PREFIX


class CustomTargetEncoder(TargetEncoder):

    def __init__(self, target, **fit_params):
        super().__init__()
        self.target = target

    def fit(self, X, y=None, **fit_params):

        return super().fit(
            X,
            y=y[self.target],
            **fit_params
        )

    def transform(self, X, y=None, **fit_params):
        # print(y[self.target])

        return super().transform(
            X,
            **fit_params
        )


# Подготовка списка признаков для передачи в mapper.
numeric_features = [[x] for x in NUM_FEATURES]
category_ohe_features = [[x] for x in CATEGORICAL_OHE_FEATURES]
category_ste_features = [[x] for x in CATEGORICAL_STE_FEATURES]

# Трансформер для вещественных признаков.
gen_numeric = gen_features(
    columns=numeric_features,
    classes=[
        {
            "class": SimpleImputer,
            "strategy": "median"
        },
        {
            "class": StandardScaler
        }
    ]
)

# Трансформер для категориальных признаков с one-hot-encoding.
gen_cat_ohe = gen_features(
    columns=category_ohe_features,
    classes=[
        {
            "class": SimpleImputer,
            "strategy": "constant"
        },
        {
            "class": OneHotEncoder
        }
    ]
)
gen_cat_ohe = [(col_name, transformer, {"alias": CAT_PREFIX + col_name[0]}) for col_name, transformer, _ in gen_cat_ohe]

# Трансформер для категориальных признаков с STE (для каждой целевой переменной).
gen_ste_tar_C = gen_features(
    columns=category_ste_features,
    classes=[
        {
            "class": SimpleImputer,
            "strategy": "constant"
        },
        {
            "class": CustomTargetEncoder,
            "target": 'C',
            "smoothing": 5
        }
    ]
)
gen_ste_tar_C = [(col_name, transformer, {"alias": CAT_PREFIX + col_name[0]}) for
                 (col_name, transformer, _) in gen_ste_tar_C]

# gen_ste_tar_TST = gen_features(
#     columns=category_ste_features,
#     classes=[
#         {
#             "class": SimpleImputer,
#             "strategy": "constant"
#         },
#         {
#             "class": SmoothedTargetEncoding,
#             "target": TARGET[1]
#         }
#     ]
# )
# gen_ste_tar_TST = [(col_name, transformer, {"alias": CAT_PREFIX + col_name[0]}) for
#                    (col_name, transformer, _) in gen_ste_tar_TST]

# Определение пайплайна через mapper.
preprocess_mapper = DataFrameMapper(
    [
        # *gen_numeric,
        # *gen_cat_ohe,
        *gen_ste_tar_C,
        # *gen_ste_tar_TST,
    ],
    input_df=True,
    df_out=True
)
# print(gen_cat_ohe)

data = pd.read_csv('C://Users//ivan.andryushin//Desktop//PyProjects//evraz//data//dummy.csv')
X_train = data[CATEGORICAL_STE_FEATURES]
y_train = data[TARGET]
print(category_ste_features)
print(preprocess_mapper.fit_transform(X_train, y_train))


"""
REFERENCES

https://contrib.scikit-learn.org/category_encoders/targetencoder.html

https://github.com/kinir/catboost-with-pipelines/blob/master/sklearn-pandas-catboost.ipynb

https://medium.com/analytics-vidhya/combining-scikit-learn-pipelines-with-catboost-and-dask-part-2-9240242966a7

"""