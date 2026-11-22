import shap
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.training import training

from utils.constants import ACTIVATIONS, BATCH_SIZES
from utils.dataset_manipulators import recursive_predict


class MLPWrapper:

    def __init__(self,
                 inputs= 0,
                 hidden_layers=0,
                 epochs=1,
                 batch_size=32,
                 hidden_activation='relu',
                 output_activation='linear',
                 optimizer='adam',
                 learning_rate=0.001,
                 loss='mse',
                 early_stopping=5,
                 random_state=42):

        self._inputs = inputs
        self._hidden_layers = hidden_layers
        self._epochs = epochs
        self._batch_size = batch_size
        self._hidden_activation = hidden_activation
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
        if self._inputs <= 0:
            return
        self._model = Sequential([
            Input(shape=(self._inputs,)),
            Dense(self._hidden_layers, activation=self._hidden_activation, use_bias=True),
            Dense(1, activation=self._output_activation)
        ])
        opt = tf.keras.optimizers.get(self._optimizer)
        opt.learning_rate = self._learning_rate
        self._model.compile(optimizer=opt, loss=self._loss, jit_compile=True)

    def fit(self, x, y, validation_split=0.2):
        self._target = x.columns

        x_2d = np.asarray(x)
        if self._model is None or self._inputs != x_2d.shape[1]:
            self._inputs = x_2d.shape[1]
            self._build_model()

        callbacks = []
        if self._early_stopping > 0 and validation_split > 0:
            callbacks.append(EarlyStopping(
                monitor='val_loss', patience=self._early_stopping, restore_best_weights=True
            ))

        self._model.fit(
            x_2d, y, epochs=self._epochs, validation_split=validation_split, batch_size=self._batch_size,
            callbacks=callbacks, verbose=0
        )
        return self

    def predict(self, x, recursive=True):
        if not recursive or len(x) <= 1:
            return self._predict_raw(x)
        return recursive_predict(self._predict_raw, x, target_name=self._target)

    def _predict_raw(self, x):
        x_2d = np.asarray(x)
        return self._model(x_2d, training=False).numpy().flatten()

    def get_importance(self, x, evaluation_split=0.2, normalize=True):
        x_arr = np.asarray(x)
        n_samples = len(x_arr)

        if evaluation_split >= 1.0:
            n_eval = int(evaluation_split)
        else:
            n_eval = int(round(n_samples * evaluation_split))

        n_eval = max(1, min(n_samples - 1, n_eval))
        split_idx = n_samples - n_eval

        bg_data = x_arr[:split_idx]
        eval_data = x_arr[split_idx:]

        explainer = shap.GradientExplainer(self._model, bg_data)
        shap_values = explainer(eval_data)
        vals = shap_values.values if hasattr(shap_values, "values") else shap_values
        importance = np.mean(np.abs(vals), axis=0).flatten()

        if normalize and np.sum(importance) > 0:
            importance = importance / np.sum(importance)

        return pd.DataFrame({
            "feature": list(x.columns),
            "importance": importance
        })

    @classmethod
    def from_array(cls, solution_arr, **fixed_params):
        act_idx = int(np.clip(round(solution_arr[3]), 0, len(ACTIVATIONS) - 1))
        bth_idx = int(np.clip(round(solution_arr[2]), 0, len(BATCH_SIZES) - 1))

        return cls(
            hidden_layers=max(1, int(round(solution_arr[0]))),
            epochs=max(1, int(round(solution_arr[1]))),
            batch_size=BATCH_SIZES[bth_idx],
            hidden_activation=ACTIVATIONS[act_idx],
            learning_rate=float(solution_arr[4]),
            **fixed_params
        )

    def to_dict(self):
        return {
            'hidden_layers': self._hidden_layers,
            'epochs': self._epochs,
            'batch_size': self._batch_size,
            'hidden_activation': self._hidden_activation,
            'learning_rate': self._learning_rate,
        }