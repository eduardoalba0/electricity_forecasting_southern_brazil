import logging
import multiprocessing
import os
import sys
import time
import warnings
import random
import concurrent.futures
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from utils.constants import (
    FEATURES_DIR,
    FEATURE_THRESHOLD_BOUND,
    HYPERPARAMS_DIR,
    PROCESSED_DATA_DIR,
    TARGETS, SEEDS, get_models, OPTIMIZERS, REQUIRED_FEATURES,
)
from wrappers import COAWrapper, MLPWrapper, PSOWrapper

warnings.filterwarnings('ignore')

def reset_seed(rnd_seed = 42):
    tf.keras.backend.clear_session()
    random.seed(rnd_seed)
    np.random.seed(rnd_seed)
    tf.random.set_seed(rnd_seed)
    tf.keras.utils.set_random_seed(rnd_seed)


os.environ["PYTHONHASHSEED"] = str(0)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.config.experimental.enable_op_determinism()

N_ITER = 10
N_SOLUTIONS = 20
FEATURE_SELECTION = True
N_SPLITS = 5
TEST_SIZE = 1
GROUPBY_COL = "State"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

def configure_gpu():
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def load_dataset(filepath = None) :
    filepath = Path(filepath) if filepath is not None else (PROCESSED_DATA_DIR / "dataset_final.csv")
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    df = pd.read_csv(filepath)
    if "Date" in df.columns:
        df = df.set_index("Date")

    df = df.dropna()

    return df


def get_feature_importance(df, features, target):
    reset_seed()
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    imp_file = FEATURES_DIR / f"{target}_feature_importance.csv"

    if imp_file.exists():
        logger.info(f"Loading feature importance file: {imp_file}")
        return pd.read_csv(imp_file)

    logger.info(f"Calculating SHAP values per state for {target}...")

    state_importances = []

    for state, df_state in df.groupby("State"):
        df_sample = df_state.iloc[:-24]

        logger.info(f"  Computing SHAP for state {state} (samples={len(df_sample)})...")
        baseline = MLPWrapper(
            inputs=len(features),
            hidden_layers=100,
            epochs=50,
            batch_size=128,
        )
        baseline.fit(df_sample[features], df_sample[target])
        imp_state = baseline.get_importance(df_sample[features], evaluation_split=12, normalize=True)
        state_importances.append(imp_state)

    if not state_importances:
        raise ValueError("Could not compute SHAP feature importances for any state.")

    df_concat = pd.concat(state_importances, ignore_index=True)
    df_mean_imp = (
        df_concat.groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values(by="importance", ascending=False)
        .reset_index(drop=True)
    )

    if df_mean_imp["importance"].sum() > 0:
        df_mean_imp["importance"] /= df_mean_imp["importance"].sum()

    df_mean_imp.to_csv(imp_file, index=False)
    logger.info(f"Mean SHAP feature importances across states saved to: {imp_file}")
    return df_mean_imp


def build_optimizer(optimizer_name, common_kwargs, n_solutions):
    opt = optimizer_name
    if opt == "PSO":
        return PSOWrapper(n_solutions=n_solutions, **common_kwargs)
    if opt == "COA":
        n_coy = 5
        n_packs = max(1, n_solutions // n_coy)
        return COAWrapper(n_packs=n_packs, n_coy=n_coy, **common_kwargs)
    raise ValueError(f"Unknown optimizer {opt}")



def run_optimization(df, features, target, model_name, model_wrapper, optimizer_name, bounds, df_importance):
    output_dir = HYPERPARAMS_DIR / target / f"{model_name}_{optimizer_name.upper()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    groupby = GROUPBY_COL if GROUPBY_COL in df.columns else None
    summary_records = []

    logger.info(
        f"Starting: Target={target} | Model={model_name} | Opt={optimizer_name.upper()} | "
        f"Seeds={len(SEEDS)} | Feature Selection={FEATURE_SELECTION}"
    )

    for idx, seed in enumerate(SEEDS, 1):
        logger.info(f"--- Seed {seed} ({idx}/{len(SEEDS)}) ---")
        reset_seed(seed)
        start_time = time.time()

        common_kwargs = dict(
            dataset=df.iloc[:-24],
            features=features,
            target=target,
            model_name=model_name,
            model_wrapper=model_wrapper,
            n_iter=N_ITER,
            bounds=bounds,
            path=output_dir,
            feature_selection=FEATURE_SELECTION,
            feature_importances=df_importance,
            required_features=REQUIRED_FEATURES,
            n_splits=N_SPLITS,
            test_size=TEST_SIZE,
            groupby_col=groupby,
            seed=seed,
        )
        optimizer = build_optimizer(optimizer_name, common_kwargs, N_SOLUTIONS)
        best_cost, _ = optimizer.optimize()
        elapsed = time.time() - start_time

        best_sol = optimizer.solutions[0] if optimizer.solutions else {}
        best_fitness = best_sol.get("fitness", best_cost)
        logger.info(f"Seed {seed} finished in {elapsed:.2f}s | Best Fitness: {best_fitness:.4f}")

        summary_records.append({
            "seed": seed,
            "target": target,
            "model": model_name,
            "optimizer": optimizer_name.upper(),
            "best_fitness": best_fitness,
            "elapsed_seconds": elapsed,
            **{k: v for k, v in best_sol.items() if k != "fitness"},
        })

    summary_path = output_dir / f"summary_{model_name}_{optimizer_name.lower()}.csv"
    pd.DataFrame(summary_records).to_csv(summary_path, sep=";", decimal=".", index=False)
    logger.info(f"The results are stored in: {summary_path}")


def optimize_target(target):
    configure_gpu()
    target_file = PROCESSED_DATA_DIR / f"dataset_final_{target.lower()}.csv"

    df_target = load_dataset(target_file)
    features_target = df_target.drop(columns=[target, "State"]).columns

    df_importance = (
        get_feature_importance(df_target, features_target, target)
        if FEATURE_SELECTION
        else None
    )

    models = get_models()
    for model in models.keys():
        model_name, model_wrapper, base_bounds = models[model]
        bounds = base_bounds + [FEATURE_THRESHOLD_BOUND] if FEATURE_SELECTION else base_bounds

        for opt_name in OPTIMIZERS:
            run_optimization(
                df=df_target,
                features=features_target,
                target=target,
                model_name=model_name,
                model_wrapper=model_wrapper,
                optimizer_name=opt_name,
                bounds=bounds,
                df_importance=df_importance,
            )


def main():

    multiprocessing.set_start_method("spawn", force=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(optimize_target, target) for target in TARGETS]

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"Process generated an exception: {exc}")


if __name__ == "__main__":
    main()