import numpy as np

from third_party.coyote_optimizer import COA
from wrappers import OptimizerBaseWrapper


class COAWrapper(OptimizerBaseWrapper):
    def __init__(self, dataset, features, target, model_name, model_wrapper,
                 n_packs, n_coy, n_iter, bounds, path, feature_selection = False, feature_importances=None, required_features=None, fixed_params=None,
                 n_splits=5, test_size=1, groupby_col=None, random_state=42):

        n_solutions = n_packs * n_coy

        super().__init__(
            dataset=dataset,
            features=features,
            target=target,
            model_name="COA-"+model_name,
            model_wrapper=model_wrapper,
            n_solutions=n_solutions,
            n_iter=n_iter,
            path=path,
            feature_selection=feature_selection,
            feature_importances=feature_importances,
            required_features=required_features,
            fixed_params=fixed_params,
            n_splits=n_splits,
            test_size=test_size,
            groupby_col=groupby_col,
            random_state=random_state
        )

        self._n_packs = n_packs
        self._n_coy = n_coy
        self._nfeval_max = self._n_solutions * self._n_iter
        self._bounds = bounds
        self._lu = None
        self._eval_count = 0
        self._build_optimizer()

    def _build_optimizer(self):
        lower, upper = zip(*self._bounds)
        self._lu = np.array([lower, upper], dtype=float)

    def optimize(self):
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

        self._solutions.sort(key=lambda s: s['fitness'])
        self._iter.append(self._solutions[0])
        self._save_csv()

        return fitness