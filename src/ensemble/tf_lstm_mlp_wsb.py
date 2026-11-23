import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Input, Lambda, LSTM, Reshape

from ensemble.tf_weaker_separator import TensorWSBagging
from utils.constants import ACTIVATIONS


class TensorLSTMMLPWSB(TensorWSBagging):
    def __init__(self,
                 inputs=None,
                 lstm_units=None,
                 lstm_activation=None,
                 n_weak_mlps=None,
                 max_mlp_hidden=None,
                 epochs=None,
                 batch_size=32,
                 optimizer="adam",
                 loss="mse",
                 learning_rate=0.0005,
                 strong_weight=0.5,
                 weak_weight=0.5,
                 random_patches=True,
                 patches_ratio=0.5,
                 early_stopping=5,
                 auto_weights=False,
                 random_state=42):

        super().__init__(
            inputs=inputs,
            epochs=epochs,
            batch_size=batch_size,
            optimizer=optimizer,
            loss=loss,
            learning_rate=learning_rate,
            strong_weight=strong_weight,
            weak_weight=weak_weight,
            random_patches=random_patches,
            patches_ratio=patches_ratio,
            early_stopping=early_stopping,
            auto_weights=auto_weights,
            random_state=random_state
        )

        self._lstm_units = lstm_units
        self._lstm_activation = lstm_activation

        self._n_weak_mlps = n_weak_mlps
        self._max_mlp_hidden = max_mlp_hidden

    def _build_model(self):
        ensemble_input = Input(shape=(self._inputs,), name="ensemble_input")
        reshaped_for_lstm = Reshape((1, self._inputs), name="reshape_lstm")(ensemble_input)
        lstm_out = LSTM(self._lstm_units, activation=self._lstm_activation, use_bias=True, name='strong_lstm')(
            reshaped_for_lstm)

        strong_out = Dense(1, activation='linear', name='strong_pred')(lstm_out)
        weak_outputs = []
        patch_size = max(1, int(self._inputs * self._patches_ratio))
        self._feature_index = []

        for i in range(self._n_weak_mlps):
            if self._random_patches:
                col_idx = self._rng.choice(self._inputs, size=patch_size, replace=False)
            else:
                col_idx = np.arange(self._inputs)
            self._feature_index.append(col_idx)

            sliced_input = Lambda(
                lambda x, idx=col_idx: tf.gather(x, idx, axis=-1),
                output_shape=(patch_size,),
                name=f'weak_patch_{i}'
            )(ensemble_input)

            flatten_layer = Flatten(name=f'weak_flat_{i}')(sliced_input)
            hidden_units = int(self._rng.integers(1, self._max_mlp_hidden, endpoint=True))
            activation = self._rng.choice(ACTIVATIONS)

            hidden = Dense(hidden_units, activation=activation, name=f'weak_hidden_{i}')(flatten_layer)
            weak_out = Dense(1, activation='linear', name=f'weak_pred_{i}')(hidden)
            weak_outputs.append(weak_out)

        self._ensemble_input = ensemble_input
        self._strong_output = strong_out
        self._weak_outputs = weak_outputs

    def _predict_raw(self, x):
        x_2d = np.asarray(x)
        if x_2d.ndim == 1:
            x_2d = x_2d.reshape(1, -1)
        elif x_2d.ndim == 3:
            x_2d = x_2d.reshape(x_2d.shape[0], -1)
        return self._model.predict(x_2d, verbose=0).flatten()

    @classmethod
    def from_array(cls, solution_arr, **fixed_params):
        act_idx = int(np.clip(round(solution_arr[1]), 0, len(ACTIVATIONS) - 1))

        return cls(
            lstm_units=max(1, int(round(solution_arr[0]))),
            lstm_activation=ACTIVATIONS[act_idx],
            n_weak_mlps=max(1, int(round(solution_arr[2]))),
            max_mlp_hidden=max(1, int(round(solution_arr[3]))),
            epochs=max(1, int(round(solution_arr[4]))),
            learning_rate=float(solution_arr[5]),
            strong_weight=float(solution_arr[6]),
            weak_weight=float(1.0 - solution_arr[6]),
            patches_ratio=float(solution_arr[7]),
            **fixed_params
        )

    def to_dict(self):
        return {
            'lstm_units': self._lstm_units,
            'lstm_activation': self._lstm_activation,
            'n_weak_mlps': self._n_weak_mlps,
            'max_mlp_hidden': self._max_mlp_hidden,
            'epochs': self._epochs,
            'learning_rate': self._learning_rate,
            'strong_weight': float(self._strong_weight.numpy()),
            'weak_weight': float(self._weak_weight.numpy()),
            'patches_ratio': self._patches_ratio,
        }