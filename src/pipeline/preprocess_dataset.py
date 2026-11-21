import os
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple
import pandas as pd

import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(BASE_DIR, "src"))

from utils.constants import STATES, QUADRANTS, CAMS_VARS, ERA5_VARS, MAX_WORKERS, TARGETS
from utils.dataset_manipulators import create_lags, quantile_outlier_removal

RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")


def process_single_weather_file(
    filepath,
    state,
    quadrant,
    variables,
):
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
        .mean()
        .reset_index()
    )

    aggregated["State"] = state
    aggregated["Quadrant"] = quadrant

    return aggregated


def _weather_worker(task):
    filepath, dataset_type, state, quadrant, variables = task
    return (
        dataset_type,
        process_single_weather_file(filepath, state, quadrant, variables),
    )


def process_all_weather_data(
    states=None,
    quadrants=None,
):
    states = states or STATES
    quadrants = quadrants or QUADRANTS

    tasks: List[Tuple[str, str, str, str, List[str]]] = []

    for state in states:
        for quad in quadrants:
            cams_file = os.path.join(RAW_DATA_DIR, "cams", f"{state}_{quad}.csv")
            era5_file = os.path.join(RAW_DATA_DIR, "era5", f"{state}_{quad}.csv")

            tasks.append((cams_file, "cams", state, quad, CAMS_VARS))
            tasks.append((era5_file, "era5", state, quad, ERA5_VARS))

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

    all_variables = CAMS_VARS + ERA5_VARS

    pivoted = merged_weather.pivot(
        index=["State", "Date"],
        columns=["Quadrant", "Hour"],
        values=all_variables,
    )

    pivoted.columns = [f"{var}_{quad}_{hour}" for var, quad, hour in pivoted.columns]
    return pivoted.reset_index()


def preprocess_consumption_data(
    consumption_filepath=os.path.join(RAW_DATA_DIR, "consumption", "consumption.csv"),
):
    if not os.path.exists(consumption_filepath):
        raise FileNotFoundError(f"Consumption file not found: {consumption_filepath}")

    df_consumption = pd.read_csv(consumption_filepath)
    df_consumption = df_consumption.drop_duplicates(subset=["Date", "State"], keep="first")
    return df_consumption


def main():
    raw_dir = RAW_DATA_DIR
    output_filepath = os.path.join(PROCESSED_DATA_DIR, "dataset.csv")
    output_cleaned_filepath = os.path.join(PROCESSED_DATA_DIR, "dataset_final.csv")

    consumption_path = os.path.join(raw_dir, "consumption", "consumption.csv")
    df_consumption = preprocess_consumption_data(consumption_path)

    df_weather = process_all_weather_data()

    df_merged = pd.merge(df_consumption, df_weather, on=["State", "Date"], how="inner")

    df_merged["_dt"] = pd.to_datetime(df_merged["Date"], format="%m/%Y")
    df_merged = df_merged.sort_values(by=["State", "_dt"]).reset_index(drop=True)

    df_merged["Month"] = df_merged["_dt"].dt.month
    df_merged["Year"] = df_merged["_dt"].dt.year

    states_included = df_merged["State"].unique().tolist()
    date_range = f"{df_merged['Date'].iloc[0]} to {df_merged['Date'].iloc[-1]}"

    df_merged = df_merged.set_index("Date")

    state_dummies = pd.get_dummies(df_merged["State"], prefix="State", dtype=int)
    month_dummies = pd.get_dummies(df_merged["Month"], prefix="Month", dtype=int)
    year_dummies = pd.get_dummies(df_merged["Year"], prefix="Year", dtype=int)

    orig_vars = df_merged[["State", "Month", "Year"]]
    df_features = df_merged.drop(columns=["State", "Month", "Year", "_dt"])

    # dataset.csv mantém as variáveis originárias dos dummies
    df_dataset = pd.concat([orig_vars, state_dummies, month_dummies, year_dummies, df_features], axis=1)

    os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
    df_dataset.to_csv(output_filepath, index=True)

    dummy_prefixes = ("State_", "Month_", "Year_")
    orig_cols = ("State", "Month", "Year")
    numeric_cols = [
        col for col in df_dataset.columns
        if col not in orig_cols and not col.startswith(dummy_prefixes)
    ]
    df_cleaned = quantile_outlier_removal(df_dataset, target=numeric_cols, groupby_col="State")
    df_cleaned = create_lags(df_cleaned, lag_col=TARGETS, n_lags=12, groupby_col="State")
    # Remove as variáveis originárias apenas de dataset_final.csv
    df_cleaned = df_cleaned.drop(columns=["State", "Month", "Year"])
    df_cleaned.to_csv(output_cleaned_filepath, index=True)

    print(f"Preprocessed dataset saved to: {output_filepath}")

    print(f"Dataset successfully created with shape: {df_dataset.shape}")
    print(f"States included: {states_included}")
    print(f"Date range: {date_range}")
    print(f"Total features: {len(df_dataset.columns)}")


if __name__ == "__main__":
    main()