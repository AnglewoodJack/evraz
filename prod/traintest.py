import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor

def traintest(df: pd.DataFrame):
    dummy_df = df
    sc = StandardScaler()

    features_c = np.array(dummy_df.iloc[:, :45])
    target_c = np.array(dummy_df.iloc[:, 45:46]) #C

    features_train_c, features_test_c, target_train_c, target_test_c = train_test_split(features_c,
                                                                                        target_c,
                                                                                        random_state=0)

    features_train_c = sc.fit_transform(features_train_c)
    features_test_c = sc.fit_transform(features_test_c)

    baseline_c = CatBoostRegressor(verbose=0, eval_metric='MAE', )  # Ridge()
    baseline_c.fit(features_train_c, target_train_c)

    features_tst = np.array(dummy_df.iloc[:, :45])
    target_tst = np.array(dummy_df.iloc[:, 46:47]) #TST

    features_train_tst, features_test_tst, target_train_tst, target_test_tst = train_test_split(features_tst,
                                                                                                target_tst,
                                                                                                random_state=0)
    features_train_tst = sc.fit_transform(features_train_tst)
    features_test_tst = sc.fit_transform(features_test_tst)

    baseline_tst = CatBoostRegressor(verbose=0, eval_metric='MAE', )  # Ridge()
    baseline_tst.fit(features_train_tst, target_train_tst)
    model_tst = baseline_tst.fit(features_train_c, target_train_c)
    model_c = baseline_c.fit(features_train_c, target_train_c)

    return (
        pd.concat([
            pd.DataFrame(model_c.predict(features_test_c)).rename(columns={0: 'C'}),
            pd.DataFrame(model_tst.predict(features_test_tst)).rename(columns={0: 'TST'})],
            axis=1),
        pd.concat([
            pd.DataFrame(target_test_c).rename(columns={0: 'C'}),
            pd.DataFrame(target_test_tst).rename(columns={0: 'TST'})],
            axis=1),
    )
