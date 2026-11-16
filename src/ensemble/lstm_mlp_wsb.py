from ensemble.weaker_separator import WSBagging
from utils.constants import BATCH_SIZES, ACTIVATIONS
from wrappers import LSTMWrapper, MLPWrapper


class LSTMMLPWSB(WSBagging):
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
                 optimizer = "adam",
                 lstm_learning_rate=0.0005,
                 mlp_learning_rate=0.001,
                 strong_weight=0.5,
                 weak_weight=0.5,
                 bootstrap=True,
                 random_patches=True,
                 bootstrap_ratio=0.5,
                 patches_ratio=0.5,
                 early_stopping=5,
                 random_state=42):

        super().__init__(
            inputs=inputs,
            timesteps=timesteps,
            strong_weight=strong_weight,
            weak_weight=weak_weight,
            bootstrap=bootstrap,
            random_patches=random_patches,
            bootstrap_ratio=bootstrap_ratio,
            patches_ratio=patches_ratio,
            early_stopping=early_stopping,
            random_state=random_state)

        self._timesteps = timesteps
        self._lstm_units = lstm_units
        self._lstm_epochs = lstm_epochs
        self._lstm_batch_size = lstm_batch_size
        self._lstm_activation = lstm_activation
        self._optimizer = optimizer
        self._lstm_learning_rate = lstm_learning_rate

        self._n_weak_mlps = max_mlps
        self._max_mlp_hidden = max_mlp_hidden
        self._max_mlp_epochs = max_mlp_epochs
        self._max_mlp_batch_size = max_mlp_batch_size
        self._mlp_learning_rate = mlp_learning_rate

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
            optimizer=self._optimizer,
            learning_rate=self._lstm_learning_rate,
            early_stopping=self._early_stopping
        )

        for _ in range(self._n_weak_mlps):
            mlp = MLPWrapper(
                inputs=max(1, int(self._inputs * self._patches_ratio)),
                hidden_layers=int(self._rng.integers(1, self._max_mlp_hidden, endpoint=True)),
                epochs=int(self._rng.integers(1, self._max_mlp_epochs, endpoint=True)),
                batch_size=self._rng.choice(BATCH_SIZES),
                hidden_activation=self._rng.choice(ACTIVATIONS),
                optimizer=self._optimizer,
                learning_rate=self._mlp_learning_rate,
                early_stopping=self._early_stopping
            )
            self._weak_predictors.append(mlp)
