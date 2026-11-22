import numpy as np

from ensemble.weaker_separator import WSBagging
from utils.constants import BATCH_SIZES, ACTIVATIONS
from wrappers import LSTMWrapper, MLPWrapper


class LSTMMLPWSB(WSBagging):
    def __init__(self,
                 inputs,
                 timesteps,
                 lstm_units,
                 lstm_activation,
                 n_weak_mlps,
                 max_mlp_hidden,
                 epochs,
                 batch_size,
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
        self._optimizer = optimizer
        self._epochs = epochs
        self._batch_size = batch_size

        self._lstm_units = lstm_units
        self._lstm_epochs = epochs
        self._lstm_activation = lstm_activation
        self._lstm_learning_rate = lstm_learning_rate

        self._n_weak_mlps = n_weak_mlps
        self._max_mlp_hidden = max_mlp_hidden
        self._mlp_learning_rate = mlp_learning_rate

        self._build_model()

    def _build_model(self):
        self._weak_predictors = []
        self._strong_predictor = LSTMWrapper(
            inputs=self._inputs,
            timesteps=self._timesteps,
            units=self._lstm_units,
            lstm_activation=self._lstm_activation,
            epochs=self._lstm_epochs,
            batch_size=self._batch_size,
            optimizer=self._optimizer,
            learning_rate=self._lstm_learning_rate,
            early_stopping=self._early_stopping
        )

        for _ in range(self._n_weak_mlps):
            mlp = MLPWrapper(
                inputs=max(1, int(self._inputs * self._patches_ratio)),
                hidden_layers=int(self._rng.integers(1, self._max_mlp_hidden, endpoint=True)),
                epochs=int(self._rng.integers(1, self._epochs, endpoint=True)),
                batch_size=self._batch_size,
                hidden_activation=self._rng.choice(ACTIVATIONS),
                optimizer=self._optimizer,
                learning_rate=self._mlp_learning_rate,
                early_stopping=self._early_stopping
            )
            self._weak_predictors.append(mlp)

    @classmethod
    def from_array(cls, solution_arr, **fixed_params):
        act_idx = int(np.clip(round(solution_arr[1]), 0, len(ACTIVATIONS) - 1))
        bth_idx = int(np.clip(round(solution_arr[5]), 0, len(BATCH_SIZES) - 1))

        return cls(
            lstm_units=max(1, int(round(solution_arr[0]))),
            lstm_activation=ACTIVATIONS[act_idx],
            n_weak_mlps=max(1, int(round(solution_arr[2]))),
            max_mlp_hidden=max(1, int(round(solution_arr[3]))),
            epochs=max(1, int(round(solution_arr[4]))),
            batch_size=BATCH_SIZES[bth_idx],
            lstm_learning_rate=float(solution_arr[6]),
            mlp_learning_rate=float(solution_arr[7]),
            strong_weight=float(solution_arr[8]),
            weak_weight=float(1.0 - solution_arr[8]),
            patches_ratio=float(solution_arr[9]),
            **fixed_params
        )

    def to_dict(self):
        return {
            'lstm_units': self._lstm_units,
            'lstm_activation': self._lstm_activation,
            'n_weak_mlps': self._n_weak_mlps,
            'max_mlp_hidden': self._max_mlp_hidden,
            'epochs': self._epochs,
            'batch_size': self._batch_size,
            'lstm_learning_rate': self._lstm_learning_rate,
            'mlp_learning_rate': self._mlp_learning_rate,
            'strong_weight': float(self._strong_weight),
            'weak_weight': float(self._weak_weight),
            'patches_ratio': self._patches_ratio,
        }