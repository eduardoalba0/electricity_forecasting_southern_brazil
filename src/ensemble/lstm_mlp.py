from ensemble.weaker_separator import WSBagging
from utils.constants import BATCH_SIZES, ACTIVATIONS
from wrappers import LSTMWrapper, MLPWrapper


class LSTMMLP(WSBagging):
    def __init__(self,
                 inputs,
                 timesteps,
                 lstm_units,
                 lstm_epochs,
                 lstm_batch_size,
                 lstm_activation,
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
            timesteps,
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

        self._timesteps = timesteps
        self._lstm_units = lstm_units
        self._lstm_epochs = lstm_epochs
        self._lstm_batch_size = lstm_batch_size
        self._lstm_activation = lstm_activation

        self._n_weak_mlps = max_mlps
        self._max_mlp_hidden = max_mlp_hidden
        self._max_mlp_epochs = max_mlp_epochs
        self._max_mlp_batch_size = max_mlp_batch_size
        self._build_model()

    def _build_model(self):
        self._weak_predictors = []
        self._strong_predictor = LSTMWrapper(
            inputs=self._inputs,
            timesteps=self._timesteps,
            units=self._lstm_units,
            epochs=self._lstm_epochs,
            lstm_activation=self._lstm_activation,
            batch_size=self._lstm_batch_size,
            early_stopping=self._early_stopping
        )

        for _ in range(self._n_weak_mlps):
            mlp = MLPWrapper(
                inputs=max(1, int(self._inputs * self._patches_ratio)),
                hidden_layers=int(self._rng.integers(1, self._max_mlp_hidden, endpoint=True)),
                epochs=int(self._rng.integers(1, self._max_mlp_epochs, endpoint=True)),
                batch_size=self._rng.choice(BATCH_SIZES),
                hidden_activation=self._rng.choice(ACTIVATIONS),
                early_stopping=self._early_stopping,
            )
            self._weak_predictors.append(mlp)

    def _align_weak_data(self, x, y=None):
        if self._timesteps > 1:
            if y is not None:
                return x[self._timesteps - 1:], y[self._timesteps - 1:]
            return x[self._timesteps - 1:]

        if y is not None:
            return x, y
        return x

    def _align_targets(self, y):
        if self._timesteps > 1:
            return y[self._timesteps - 1:]
        return y