from typing import List
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper, gen_features
from prod.transformers import CustomTargetEncoder


def pipeline_mapper(numerical: List[str], ohe_categorical: List[str],
                    ste_categorical: List[str], targets: List[str], prefix: str) -> DataFrameMapper:
    """
    Создает пайплайн для обработки данных. Эстиматор сюда не включается.
    :param prefix: префикс для новых названий обработанных с помощью энкодеров категориальных признаков.
    :param targets:
    :param numerical:
    :param ohe_categorical:
    :param ste_categorical:
    :return:
    """
#    Подготовка списка признаков для передачи в mapper.
    numeric_features = [[x] for x in numerical]
    category_ohe_features = [[x] for x in ohe_categorical]
    category_ste_features = [[x] for x in ste_categorical]
    # Трансформер для вещественных признаков.
    gen_numeric = gen_features(
        columns=numeric_features,
        classes=[
            {
                "class": SimpleImputer, # Трансформер.
                "strategy": "median"    # Параметры.
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
    gen_cat_ohe = [(col_name, transformer, {"alias": prefix + col_name[0]}) for
                   (col_name, transformer, _) in gen_cat_ohe]
    # Трансформер для категориальных признаков с STE (первая целевая переменная).
    gen_ste_tar_1 = gen_features(
        columns=category_ste_features,
        classes=[
            {
                "class": SimpleImputer,
                "strategy": "constant"
            },
            {
                "class": CustomTargetEncoder,
                "target": targets[0]
            }
        ]
    )
    gen_ste_tar_1 = [(col_name, transformer, {"alias": prefix + targets[0] + '_'  + col_name[0]}) for
                     (col_name, transformer, _) in gen_ste_tar_1]
    # Трансформер для категориальных признаков с STE (вторая целевая переменная).
    gen_ste_tar_2 = gen_features(
        columns=category_ste_features,
        classes=[
            {
                "class": SimpleImputer,
                "strategy": "constant"
            },
            {
                "class": CustomTargetEncoder,
                "target": targets[1]
            }
        ]
    )
    gen_ste_tar_2 = [(col_name, transformer, {"alias": prefix + targets[1] + '_' + col_name[0]}) for
                       (col_name, transformer, _) in gen_ste_tar_2]

    # Определение пайплайна через mapper.
    preprocess_mapper = DataFrameMapper(
        [
            *gen_numeric,
            *gen_cat_ohe,
            *gen_ste_tar_1,
            *gen_ste_tar_2,
        ],
        input_df=True,
        df_out=True
    )
    return preprocess_mapper
