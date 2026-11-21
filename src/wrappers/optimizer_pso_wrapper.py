import numpy as np
from pyswarms.single import GlobalBestPSO

from wrappers import OptimizerBaseWrapper


class PSOWrapper(OptimizerBaseWrapper):
    def __init__(self, dataset, features, target, model_name, model_wrapper,
                 n_solutions, n_iter, bounds, path, fixed_params=None,
                 options=None, n_splits=5, test_size=1, groupby_col=None, seed=42):

        super().__init__(
            dataset=dataset,
            features=features,
            target=target,
            model_name=model_name,
            model_wrapper=model_wrapper,
            n_solutions=n_solutions,
            n_iter=n_iter,
            path=path,
            fixed_params=fixed_params,
            n_splits=n_splits,
            test_size=test_size,
            groupby_col=groupby_col,
            seed=seed
        )

        self._options = options if options is not None else {'c1': 0.5, 'c2': 0.5, 'w': 0.5}
        self._bounds = bounds
        self._optimizer = None
        self._build_optimizer()

    def _build_optimizer(self):
        lower, upper = zip(*self._bounds)
        pso_bounds = (np.array(lower, dtype=float), np.array(upper, dtype=float))

        self._optimizer = GlobalBestPSO(
            n_particles=self.n_solutions,
            dimensions=len(lower),
            options=self._options,
            bounds=pso_bounds
        )

    def _optimize(self):
        return self._optimizer.optimize(self._run, iters=self.n_iter)

    def _run(self, parts):
        fit_lst = [self._run_objective_function(parts[j]) for j in range(self.n_solutions)]

        self.solutions.sort(key=lambda s: s['fitness'])
        self.iter.append(self.solutions[0])
        self._save_csv()

        return np.array(fit_lst)