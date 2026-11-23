import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple

import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(BASE_DIR, "src"))

from utils.dataset_manipulators import quantile_outlier_inputation, create_lags
from utils.constants import STATES, QUADRANTS, CAMS_VARS, ERA5_VARS, MAX_WORKERS, TARGETS, WEATHER_AGG_STATS, \
    CONSUMERS_VARS

RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

def process_single_weather_file(
    filepath,
    state,
    quadrant,
    variables,
    agg_stats=None,
):
    if agg_stats is None:
        agg_stats = WEATHER_AGG_STATS

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Weather file not found: {filepath}")

    usecols = ["time"] + variables
    dtypes = {var: "float32" for var in variables}

    df = pd.read_csv(filepath, usecols=usecols, dtype=dtypes)

    dt = pd.to_datetime(df["time"], format="mixed")
    date_key = dt.dt.strftime("%m/%Y")
    hour_key = dt.dt.strftime("%H%M")

    aggregated = (
        df[variables]
        .groupby([date_key.rename("Date"), hour_key.rename("Hour")], sort=False)
        .agg(agg_stats)
    )

    aggregated.columns = [f"{var}::{stat}" for var, stat in aggregated.columns]
    aggregated = aggregated.reset_index()

    aggregated["State"] = state
    aggregated["Quadrant"] = quadrant

    return aggregated


def _weather_worker(task):
    filepath, dataset_type, state, quadrant, variables, agg_stats = task
    return (
        dataset_type,
        process_single_weather_file(filepath, state, quadrant, variables, agg_stats=agg_stats),
    )


def process_all_weather_data(
    states=None,
    quadrants=None,
    agg_stats=None,
):
    states = states or STATES
    quadrants = quadrants or QUADRANTS
    agg_stats = agg_stats or WEATHER_AGG_STATS

    tasks: List[Tuple[str, str, str, str, List[str], List[str]]] = []

    for state in states:
        for quad in quadrants:
            cams_file = os.path.join(RAW_DATA_DIR, "cams", f"{state}_{quad}.csv")
            era5_file = os.path.join(RAW_DATA_DIR, "era5", f"{state}_{quad}.csv")

            tasks.append((cams_file, "cams", state, quad, CAMS_VARS, agg_stats))
            tasks.append((era5_file, "era5", state, quad, ERA5_VARS, agg_stats))

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(_weather_worker, tasks))

    cams_dfs = [df for dtype, df in results if dtype == "cams"]
    era5_dfs = [df for dtype, df in results if dtype == "era5"]

    df_cams = pd.concat(cams_dfs, ignore_index=True)
    df_era5 = pd.concat(era5_dfs, ignore_index=True)

    merged_weather = pd.merge(
        df_cams,
        df_era5,
        on=["State", "Date", "Quadrant", "Hour"],
        how="inner",
    )

    weather_stat_vars = [c for c in merged_weather.columns if "::" in c]

    pivoted = merged_weather.pivot(
        index=["State", "Date"],
        columns=["Quadrant", "Hour"],
        values=weather_stat_vars,
    )

    pivoted.columns = [
        f"{col.split('::')[0]}_{quad}_{hour}_{col.split('::')[1]}"
        for col, quad, hour in pivoted.columns
    ]
    return pivoted.reset_index()


def preprocess_consumption_data(
    consumption_filepath=os.path.join(RAW_DATA_DIR, "consumption", "consumption.csv"),
):
    if not os.path.exists(consumption_filepath):
        raise FileNotFoundError(f"Consumption file not found: {consumption_filepath}")

    df_consumption = pd.read_csv(consumption_filepath)
    return df_consumption


def main():
    raw_dir = RAW_DATA_DIR

    consumption_path = os.path.join(raw_dir, "consumption", "consumption.csv")
    df_consumption = preprocess_consumption_data(consumption_path)

    df_weather = process_all_weather_data()

    df_merged = pd.merge(df_consumption, df_weather, on=["State", "Date"], how="inner")

    df_merged["_dt"] = pd.to_datetime(df_merged["Date"], format="%m/%Y")
    df_merged = df_merged.sort_values(by=["State", "_dt"]).reset_index(drop=True)

    df_merged["Month"] = df_merged["_dt"].dt.month

    states_included = df_merged["State"].unique().tolist()
    date_range = f"{df_merged['Date'].iloc[0]} to {df_merged['Date'].iloc[-1]}"

    df_merged = df_merged.set_index("Date")

    state_dummies = pd.get_dummies(df_merged["State"], prefix="State", dtype=int)
    month_dummies = pd.get_dummies(df_merged["Month"], prefix="Month", dtype=int)

    orig_vars = df_merged[["State", "Month"]]
    df_features = df_merged.drop(columns=["State", "Month", "_dt"])

    df_dataset = pd.concat([orig_vars, state_dummies, month_dummies, df_features], axis=1)

    zero_var_cols = [c for c in df_dataset.select_dtypes(include="number").columns if df_dataset[c].std() == 0]
    if zero_var_cols:
        df_dataset = df_dataset.drop(columns=zero_var_cols)

    weather_prefixes = tuple(ERA5_VARS) + tuple(CAMS_VARS)
    numeric_cols = [
        col for col in CONSUMERS_VARS + [
            c for c in df_dataset.columns if c.startswith(weather_prefixes)
        ]
        if col in df_dataset.columns
    ]

    df_cleaned = quantile_outlier_inputation(df_dataset, target=numeric_cols)

    zero_var_cleaned = [c for c in df_cleaned.select_dtypes(include="number").columns if df_cleaned[c].std() == 0]
    if zero_var_cleaned:
        df_cleaned = df_cleaned.drop(columns=zero_var_cleaned)

    for target in TARGETS:
        target_slug = target.lower()
        consumer_col = f"{target} Consumers"
        target_lag_cols = [c for c in [target, consumer_col] if c in df_cleaned.columns]

        df_target_raw = create_lags(df_dataset, lag_col=target_lag_cols, n_lags=12, groupby_col="State")
        df_target_clean = create_lags(df_cleaned, lag_col=target_lag_cols, n_lags=12, groupby_col="State")

        other_targets = [t for t in TARGETS if t != target]
        other_consumers = [c for c in CONSUMERS_VARS if c != consumer_col]
        cols_to_drop = [c for c in other_targets + other_consumers + ["Month"] if c in df_target_raw.columns]

        df_target_raw = df_target_raw.drop(columns=cols_to_drop)
        df_target_clean = df_target_clean.drop(columns=cols_to_drop)

        target_output_path = os.path.join(PROCESSED_DATA_DIR, f"dataset_{target_slug}.csv")
        target_final_output_path = os.path.join(PROCESSED_DATA_DIR, f"dataset_final_{target_slug}.csv")

        df_target_raw.to_csv(target_output_path, index=True)
        df_target_clean.to_csv(target_final_output_path, index=True)

if __name__ == "__main__":
    main()