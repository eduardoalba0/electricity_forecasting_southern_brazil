from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit

from utils.measures import rrmse


class OptimizerBaseWrapper:
    def __init__(self, dataset, features, target, model_name, model_wrapper,
                 n_solutions, n_iter, path, fixed_params=None,
                 n_splits=5, test_size=1, groupby_col=None, seed=42):

        self.dataset = dataset
        self.features = features
        self.target = target
        self.model_name = model_name
        self.model_wrapper = model_wrapper
        self.n_solutions = n_solutions
        self.n_iter = n_iter
        self.path = Path(path)

        self.fixed_params = fixed_params if fixed_params is not None else {}
        self.n_splits = n_splits
        self.test_size = test_size
        self.groupby_col = groupby_col

        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.solutions = []
        self.iter = []

    def _build_optimizer(self):
        raise NotImplementedError

    def _optimize(self):
        raise NotImplementedError

    def _run_objective_function(self, particle):
        csv_all = []
        model_params = None

        groups = (
            self.dataset.groupby(self.groupby_col)
            if self.groupby_col is not None
            else [(None, self.dataset)]
        )

        for _, data in groups:
            pred = []
            true = []

            x = data[self.features]
            y = data[self.target]

            # TODO Impl. recursive prediction
            tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=self.test_size)
            for i_train, i_test in tscv.split(data):
                x_train, y_train = x.iloc[i_train], y.iloc[i_train]
                x_test, y_test = x.iloc[i_test], y.iloc[i_test]

                tf.keras.backend.clear_session()
                model = self.model_wrapper.from_particle(particle, **self.fixed_params)
                if model_params is None:
                    model_params = model.to_dict()

                model.fit(x_train, y_train)

                y_pred = model.predict(x_test)
                pred.extend(np.ravel(y_pred))
                true.extend(np.ravel(y_test))

            csv_all.append(rrmse(np.array(true), np.array(pred)).mean())

        fitness = float(np.mean(csv_all))
        self.solutions.append({'fitness': fitness, **model_params})
        return fitness

    def _save_csv(self):
        self.path.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.iter)
        df.to_csv(self.path / f"{self.model_name} SEED {self.seed}.csv", sep=";", decimal=".", index=False)