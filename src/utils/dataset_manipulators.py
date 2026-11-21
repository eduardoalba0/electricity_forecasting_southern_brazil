import pandas as pd

def create_lags(dataset, lag_col, n_lags, groupby_col=None):
    cols = [lag_col] if isinstance(lag_col, str) else list(lag_col)
    source = dataset.groupby(groupby_col)[cols] if groupby_col is not None else dataset[cols]

    lag_dfs = [
        source.shift(i).add_suffix(f"_LAG_{i:02d}")
        for i in range(1, n_lags + 1)
    ]

    return pd.concat([dataset] + lag_dfs, axis=1)

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
