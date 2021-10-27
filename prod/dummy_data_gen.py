import pandas as pd
import numpy as np

def _oh(value):
    if value < 0.5:
        return 0
    else:
        return 1

def generate_dummy_df(n_rows: int, n_columns: int, num_oh: int, num_ts: int):
    res = pd.DataFrame(np.random.rand(n_rows, n_columns + 2))
    if num_oh > 0:
        for col in [int(x) for x in res.columns[-num_oh-2:-2]]:
            res[col] = res[col].apply(lambda x: _oh(x))
    if num_ts > 0:
        res['date'] = pd.date_range(start='1/1/1979', periods=n_rows, freq='D')

    return res.rename(columns={n_columns: 'C', n_columns+1: "TST"}).to_csv('dummy.csv', index=False)

