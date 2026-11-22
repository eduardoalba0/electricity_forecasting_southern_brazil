import shap
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.callbacks import EarlyStopping

from utils.constants import ACTIVATIONS, BATCH_SIZES
from utils.dataset_manipulators import recursive_predict


class LSTMWrapper:
    def __init__(self,
                 inputs,
                 timesteps,
                 units,
                 epochs=1,
                 batch_size=32,
                 lstm_activation='relu',
                 output_activation='linear',
                 optimizer='adam',
                 learning_rate=0.001,
                 loss='mse',
                 early_stopping=5,
                 random_state=42):

        self._inputs = inputs
        self._units = units
        self._epochs = epochs
        self._batch_size = batch_size
        self._timesteps = timesteps
        self._lstm_activation = lstm_activation
        self._output_activation = output_activation
        self._optimizer = optimizer
        self._learning_rate = learning_rate
        self._loss = loss
        self._early_stopping = early_stopping
        self._rng = np.random.default_rng(random_state)

        self._target = None
        self._model = None
        self._build_model()

    def _build_model(self):
        self._model = Sequential([
            Input(shape=(self._timesteps, self._inputs)),
            LSTM(self._units, activation=self._lstm_activation, use_bias=True),
            Dense(1, activation=self._output_activation)
        ])

        opt = tf.keras.optimizers.get(self._optimizer)
        opt.learning_rate = self._learning_rate
        self._model.compile(optimizer=opt, loss=self._loss, jit_compile=True)

    def fit(self, x, y, validation_split=0.2):
        self._target = getattr(y, "name", None) or getattr(y, "columns", None)

        x_2d = np.asarray(x)
        y_arr = np.asarray(y)

        x_3d = np.reshape(x_2d, (x_2d.shape[0], self._timesteps, -1))

        callbacks = []
        if self._early_stopping > 0 and validation_split > 0:
            callbacks.append(EarlyStopping(
                monitor='val_loss', patience=self._early_stopping, restore_best_weights=True
            ))

        self._model.fit(
            x_3d, y_arr, epochs=self._epochs, validation_split=validation_split, batch_size=self._batch_size,
            callbacks=callbacks, verbose=0
        )
        return self

    def predict(self, x, recursive=True):
        if not recursive or len(x) <= 1:
            return self._predict_raw(x)
        return recursive_predict(self._predict_raw, x, target_name=self._target)

    def _predict_raw(self, x):
        x_2d = np.asarray(x)
        x_3d = np.reshape(x_2d, (x_2d.shape[0], self._timesteps, -1))
        return self._model(x_3d, training=False).numpy().flatten()

    def get_importance(self, x, evaluation_split=0.2, normalize=True):
        x_arr = np.asarray(x)
        n_samples = len(x_arr)

        if evaluation_split >= 1.0:
            n_eval = int(evaluation_split)
        else:
            n_eval = int(round(n_samples * evaluation_split))

        n_eval = max(1, min(n_samples - 1, n_eval))
        split_idx = n_samples - n_eval

        bg_3d = np.reshape(x_arr[:split_idx], (split_idx, self._timesteps, -1))
        eval_3d = np.reshape(x_arr[split_idx:], (n_eval, self._timesteps, -1))

        explainer = shap.GradientExplainer(self._model, bg_3d)
        shap_values = explainer.shap_values(eval_3d)
        vals = shap_values.values if hasattr(shap_values, "values") else shap_values
        importance = np.mean(np.abs(vals), axis=(0, 1)).flatten()

        if normalize and np.sum(importance) > 0:
            importance = importance / np.sum(importance)

        if hasattr(x, "columns"):
            feature_names = list(x.columns)
        else:
            feature_names = [f"feature_{i}" for i in range(x_arr.shape[1])]

        return pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        })

    @classmethod
    def from_array(cls, solution_arr, **fixed_params):
        act_idx = int(np.clip(round(solution_arr[3]), 0, len(ACTIVATIONS) - 1))
        bth_idx = int(np.clip(round(solution_arr[2]), 0, len(BATCH_SIZES) - 1))

        return cls(
            units=max(1, int(round(solution_arr[0]))),
            epochs=max(1, int(round(solution_arr[1]))),
            batch_size=BATCH_SIZES[bth_idx],
            lstm_activation=ACTIVATIONS[act_idx],
            learning_rate=float(solution_arr[4]),
            **fixed_params
        )

    def to_dict(self):
        return {
            'units': self._units,
            'epochs': self._epochs,
            'batch_size': self._batch_size,
            'lstm_activation': self._lstm_activation,
            'learning_rate': self._learning_rate,
        }