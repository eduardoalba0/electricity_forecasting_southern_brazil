from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import delayed, Parallel
from sklearn.model_selection import TimeSeriesSplit

from utils.cache import _build_cache_key
from utils.constants import N_JOBS
from utils.delayed_utils import eval_single_fold
from utils.measures import rrmse

class OptimizerBaseWrapper:
    def __init__(self, dataset, features, target, model_name, model_wrapper,
                 n_solutions, n_iter, path, feature_selection=False,
                 feature_importances=None, required_features=None,
                 fixed_params=None, n_splits=5, test_size=1,
                 groupby_col=None, seed=42):

        if feature_selection and feature_importances is None:
            raise ValueError("feature_importances cannot be None when feature_selection is True.")

        self._dataset = dataset
        self._features = list(features)
        self._target = target
        self._model_name = model_name
        self._model_wrapper = model_wrapper
        self._n_solutions = n_solutions
        self._n_iter = n_iter
        self._path = Path(path)

        self._feature_selection = feature_selection
        self._required_features = set(required_features) if required_features is not None else set()
        self.feature_importances = (
            self._prepare_importances(feature_importances)
            if feature_importances is not None
            else None
        )

        self.fixed_params = fixed_params if fixed_params is not None else {}
        self.n_splits = n_splits
        self.test_size = test_size
        self.groupby_col = groupby_col

        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.solutions = []
        self.iter = []
        self._cache = {}

    def _build_optimizer(self):
        raise NotImplementedError

    def _optimize(self):
        raise NotImplementedError

    def optimize(self):
        return self._optimize()

    def _select_features(self, threshold):
        df_imp = self.feature_importances
        selected = set(df_imp.loc[df_imp["importance_norm"] >= threshold, "feature"])
        selected.update(self._required_features)

        if not selected:
            selected.add(df_imp.iloc[0]["feature"])

        return [col for col in self._features if col in selected]

    def _run_objective_function(self, solution_arr):
        if self._feature_selection:
            model_arr = solution_arr[:-1]
            threshold = float(np.clip(solution_arr[-1], 0.0, 1.0))
            features = self._select_features(threshold)
        else:
            model_arr = solution_arr
            threshold = None
            features = self._features

        base_model = self._model_wrapper.from_array(model_arr, **self.fixed_params)
        model_params = base_model.to_dict()

        key = _build_cache_key(model_params, features)
        if key in self._cache:
            cached_solution = self._cache[key].copy()
            cached_solution["threshold"] = threshold
            self.solutions.append(cached_solution)
            return cached_solution["fitness"]

        csv_all = []
        dataset = self._dataset

        groups = (
            dataset.groupby(self.groupby_col)
            if self.groupby_col is not None
            else [(None, dataset)]
        )

        for _, data in groups:
            x = data[features]
            y = data[self._target]

            tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=self.test_size)
            splits = list(tscv.split(data))

            fold_results = Parallel(n_jobs=min(self.n_splits, N_JOBS), prefer="threads")(
                delayed(eval_single_fold)(
                    self._model_wrapper, model_arr, self.fixed_params,
                    x.iloc[i_train], y.iloc[i_train],
                    x.iloc[i_test], y.iloc[i_test]
                )
                for i_train, i_test in splits
            )

            pred = [p for res in fold_results for p in res[0]]
            true = [t for res in fold_results for t in res[1]]

            pred_arr = np.array(pred)
            true_arr = np.array(true)
            if np.isnan(pred_arr).any() or np.isinf(pred_arr).any():
                csv_all.append(1e6)
            else:
                csv_all.append(rrmse(true_arr, pred_arr).mean())

        fitness = float(np.mean(csv_all))
        if np.isnan(fitness) or np.isinf(fitness):
            fitness = 1e6
        solution_record = {
            "fitness": fitness,
            **model_params,
            "threshold": threshold,
            "n_features": len(features),
        }
        self._cache[key] = solution_record
        self.solutions.append(solution_record.copy())
        return fitness

    def _prepare_importances(self, df_imp):
        df = df_imp.sort_values(by="importance", ascending=False).reset_index(drop=True).copy()
        max_imp = df["importance"].max()
        df["importance_norm"] = df["importance"] / max_imp if max_imp > 0 else 0.0
        return df

    @property
    def cache(self):
        return self._cache


    def _save_csv(self):
        self._path.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.iter)
        df.to_csv(self._path / f"{self._model_name} SEED {self.seed}.csv", sep=";", decimal=".", index=False)

