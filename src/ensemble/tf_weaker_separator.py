import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Add, Average, Lambda, Dense
from tensorflow.keras.models import Model

class TensorWSBagging:
    def __init__(self,
                 inputs,
                 timesteps,
                 epochs=1,
                 batch_size=32,
                 optimizer='adam',
                 loss='mse',
                 learning_rate=0.0005,
                 strong_weight=0.5,
                 weak_weight=0.5,
                 random_patches=True,
                 patches_ratio=0.5,
                 early_stopping=5,
                 random_state=42):

        if not (0.0 < patches_ratio <= 1.0):
            raise ValueError("Ratios must be in (0.0, 1.0] interval.")

        self._inputs = inputs
        self._strong_predictor = None
        self._weak_predictors = []
        self._strong_weight = tf.constant(strong_weight, dtype=tf.float32)
        self._weak_weight = tf.constant(weak_weight, dtype=tf.float32)
        self._epochs = epochs
        self._batch_size = batch_size
        self._optimizer = optimizer
        self._learning_rate = float(learning_rate)
        self._loss = loss

        self._random_patches = random_patches
        self._patches_ratio = patches_ratio
        self._early_stopping = early_stopping
        self._timesteps = timesteps

        self._random_state = random_state
        self._rng = np.random.default_rng(self._random_state)
        self._feature_index = []

        self._model = None
        self._ensemble_input = None
        self._strong_output = None
        self._weak_outputs = []

    def _build_model(self):
        raise NotImplementedError

    def _compile_model(self):
        if len(self._weak_outputs) > 0:
            avg_weak = Average(name='avg_weak')(self._weak_outputs)

            if self._strong_output is not None:
                weight_sum = self._strong_weight + self._weak_weight
                w_strong = self._strong_weight / weight_sum
                w_weak = self._weak_weight / weight_sum

                weighted_strong = self._strong_output * w_strong
                weighted_weak = avg_weak * w_weak

                final_output = Add(name="ensemble_weights")([weighted_strong, weighted_weak])
            else:
                final_output = avg_weak

        else:
            final_output = self._strong_output

        self._model = Model(inputs=self._ensemble_input, outputs=final_output)

        opt = tf.keras.optimizers.get(self._optimizer)
        opt.learning_rate = self._learning_rate
        self._model.compile(optimizer=opt, loss=self._loss)


    def fit(self, x, y, validation_split=0.2):

        if self._model is None:
            self._build_model()
            self._compile_model()

        x_arr = np.asarray(x)
        y_arr = np.asarray(y)

        callbacks = []
        if self._early_stopping > 0 and validation_split > 0:
            callbacks.append(EarlyStopping(
                monitor='val_loss', patience=self._early_stopping, restore_best_weights=True
            ))


        return self._model.fit(
            x=x_arr,
            y=y_arr,
            validation_split=validation_split,
            epochs=self._epochs,
            batch_size=self._batch_size,
            callbacks=callbacks,
            verbose=0
        )

    def predict(self, x):
        x_arr = np.asarray(x)
        return self._model.predict(x=x_arr, verbose=0).flatten()