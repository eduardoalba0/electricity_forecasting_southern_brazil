from ensemble.weaker_separator import WSBagging
from utils.constants import BATCH_SIZES, ACTIVATIONS
from wrappers import MLPWrapper


class MLP(WSBagging):
    def __init__(self,
                 inputs,
                 mlp_strong_hidden,
                 mlp_strong_epochs,
                 mlp_strong_batch_size,
                 mlp_strong_activation,
                 max_mlps,
                 max_mlp_hidden,
                 max_mlp_epochs,
                 max_mlp_batch_size,
                 strong_weight=0.5,
                 weak_weight=0.5,
                 bootstrap=True,
                 random_patches=True,
                 bootstrap_ratio=0.5,
                 patches_ratio=0.5,
                 early_stopping=5,
                 random_state=42):

        super().__init__(
            inputs,
            1,
            None,
            None,
            strong_weight,
            weak_weight,
            bootstrap,
            random_patches,
            bootstrap_ratio,
            patches_ratio,
            early_stopping,
            random_state)

        self._mlp_strong_hidden = mlp_strong_hidden
        self._mlp_strong_epochs = mlp_strong_epochs
        self._mlp_strong_batch_size = mlp_strong_batch_size
        self._mlp_strong_activation = mlp_strong_activation

        self._n_weak_mlps = max_mlps
        self._max_mlp_hidden = max_mlp_hidden
        self._max_mlp_epochs = max_mlp_epochs
        self._max_mlp_batch_size = max_mlp_batch_size
        self._build_model()

    def _build_model(self):
        self._weak_predictors = []
        self._strong_predictor = MLPWrapper(
                inputs=self._inputs,
                hidden_layers=self._mlp_strong_hidden,
                epochs=self._mlp_strong_epochs,
                batch_size=self._mlp_strong_batch_size,
                hidden_activation=self._mlp_strong_activation,
                early_stopping=self._early_stopping
        )
        for _ in range(self._n_weak_mlps):
            mlp = MLPWrapper(
                inputs=max(1, int(self._inputs * self._patches_ratio)),
                hidden_layers=int(self._rng.integers(1, self._max_mlp_hidden, endpoint=True)),
                epochs=int(self._rng.integers(1, self._max_mlp_epochs, endpoint=True)),
                batch_size=self._rng.choice(BATCH_SIZES),
                hidden_activation=self._rng.choice(ACTIVATIONS),
                early_stopping=self._early_stopping
            )
            self._weak_predictors.append(mlp)