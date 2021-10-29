import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper, gen_features

from prod.data_transformers import CustomTargetEncoder
from prod.settings import CATEGORICAL_OHE_FEATURES, CATEGORICAL_STE_FEATURES, NUM_FEATURES, TARGET, CAT_PREFIX


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
            "target": TARGET[0],
            "smoothing": 5
        }
    ]
)
gen_ste_tar_C = [(col_name, transformer, {"alias": CAT_PREFIX + TARGET[0] + col_name[0]}) for
                 (col_name, transformer, _) in gen_ste_tar_C]

gen_ste_tar_TST = gen_features(
    columns=category_ste_features,
    classes=[
        {
            "class": SimpleImputer,
            "strategy": "constant"
        },
        {
            "class": CustomTargetEncoder,
            "target": TARGET[1],
            "smoothing": 5
        }
    ]
)
gen_ste_tar_TST = [(col_name, transformer, {"alias": CAT_PREFIX + TARGET[1] + col_name[0]}) for
                   (col_name, transformer, _) in gen_ste_tar_TST]

# Определение пайплайна через mapper.
preprocess_mapper = DataFrameMapper(
    [
        *gen_numeric,
        *gen_cat_ohe,
        *gen_ste_tar_C,
        *gen_ste_tar_TST,
    ],
    input_df=True,
    df_out=True
)



"""
REFERENCES

https://contrib.scikit-learn.org/category_encoders/targetencoder.html

https://github.com/kinir/catboost-with-pipelines/blob/master/sklearn-pandas-catboost.ipynb

https://medium.com/analytics-vidhya/combining-scikit-learn-pipelines-with-catboost-and-dask-part-2-9240242966a7

"""