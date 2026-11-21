import numpy as np

from third_party.coyote_optimizer import COA
from wrappers import OptimizerBaseWrapper


class COAWrapper(OptimizerBaseWrapper):
    def __init__(self, dataset, features, target, model_name, model_wrapper,
                 n_packs, n_coy, n_iter, bounds, path, fixed_params=None,
                 n_splits=5, test_size=1, groupby_col=None, seed=42):

        n_solutions = n_packs * n_coy

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

        self._n_packs = n_packs
        self._n_coy = n_coy
        self._nfeval_max = self.n_solutions * self.n_iter
        self._bounds = bounds
        self._lu = None
        self._eval_count = 0
        self._build_optimizer()

    def _build_optimizer(self):
        lower, upper = zip(*self._bounds)
        self._lu = np.array([lower, upper], dtype=float)

    def _optimize(self):
        self._eval_count = 0

        best_cost, best_pos = COA(
            FOBJ=self._run,
            lu=self._lu,
            nfevalMAX=self._nfeval_max,
            n_packs=self._n_packs,
            n_coy=self._n_coy
        )
        return best_cost, best_pos

    def _run(self, coyote):
        fitness = self._run_objective_function(coyote)
        self._eval_count += 1

        self.solutions.sort(key=lambda s: s['fitness'])
        self.iter.append(self.solutions[0])
        self._save_csv()

        return fitness