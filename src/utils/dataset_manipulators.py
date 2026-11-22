import re
import numpy as np
import pandas as pd

from utils.constants import CONSUMERS_VARS, CONSUMERS_SLUG


def create_lags(dataset, lag_col, n_lags, groupby_col=None):
    cols = [lag_col] if isinstance(lag_col, str) else list(lag_col)
    source = dataset.groupby(groupby_col)[cols] if groupby_col is not None else dataset[cols]

    lag_dfs = [
        source.shift(i).add_suffix(f"_LAG_{i:02d}")
        for i in range(1, n_lags + 1)
    ]
    df = pd.concat([dataset] + lag_dfs, axis=1)
    df = df.dropna()
    return df

def merge_datasets(dfs):
    df_merged = None
    for df in dfs.values():
        if df_merged is None:
            df_merged = df.copy()
        else:
            new_cols = [c for c in df.columns if c not in df_merged.columns]
            df_merged = pd.concat([df_merged, df[new_cols]], axis=1)
    return df_merged

def quantile_outlier_inputation(dataset, target, groupby_col=None, factor=1.5):
    df = dataset.copy()
    cols = [target] if isinstance(target, str) else list(target)

    if groupby_col is not None:
        grouped = df.groupby(groupby_col)[cols]
        q1 = grouped.transform(lambda s: s.quantile(0.25))
        q3 = grouped.transform(lambda s: s.quantile(0.75))
    else:
        q1 = df[cols].quantile(0.25)
        q3 = df[cols].quantile(0.75)

    iqr = q3 - q1
    lim_inf = q1 - factor * iqr
    lim_sup = q3 + factor * iqr

    df[cols] = df[cols].clip(lower=lim_inf, upper=lim_sup, axis=1)
    return df


def recursive_predict(predict_fn, x, target_name):
    if len(x) <= 1 or not hasattr(x, "columns"):
        return predict_fn(x)

    cols = list(x.columns)
    df_vals = x.copy()

    lags_by_prefix = {}
    for col in cols:
        match = re.search(r"^(.*)_LAG_(\d+)$", str(col))
        if match:
            prefix = match.group(1)
            lag_idx = int(match.group(2))
            lags_by_prefix.setdefault(prefix, {})[lag_idx] = col

    if not lags_by_prefix:
        return predict_fn(x)

    if target_name in lags_by_prefix:
        lag_dict = lags_by_prefix[target_name]
    else:
        lag_dict = next(iter(lags_by_prefix.values()))

    col_to_idx = {c: i for i, c in enumerate(cols)}
    lag_col_indices = {lag_i: col_to_idx[col_name] for lag_i, col_name in lag_dict.items()}
    vals = df_vals.to_numpy(copy=True)

    preds = []
    for t in range(len(vals)):
        if t > 0:
            for lag_idx, col_idx in lag_col_indices.items():
                if t >= lag_idx:
                    vals[t, col_idx] = preds[t - lag_idx]
                else:
                    prev_lag_idx = lag_idx - t
                    if prev_lag_idx in lag_col_indices:
                        prev_col_idx = lag_col_indices[prev_lag_idx]
                        vals[t, col_idx] = vals[0, prev_col_idx]

        row_t = pd.DataFrame(vals[[t]], columns=cols)
        pred_t = predict_fn(row_t)
        preds.append(float(np.ravel(pred_t)[0]))

    return np.array(preds)

