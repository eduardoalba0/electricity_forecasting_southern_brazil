import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

class MLPWrapper:
    def __init__(self,
                 inputs,
                 hidden_layers,
                 epochs=1,
                 batch_size=32,
                 hidden_activation='relu',
                 output_activation='linear',
                 optimizer='adam',
                 learning_rate=0.001,
                 loss='mse',
                 early_stopping=5):

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
        self._model = None
        self._build_model()

    def _build_model(self):
        self._model = Sequential([
            Input(shape=(self._inputs,)),
            Dense(self._hidden_layers, activation=self._hidden_activation),
            Dense(1, activation=self._output_activation)
        ])
        opt = tf.keras.optimizers.get(self._optimizer)
        opt.learning_rate = self._learning_rate
        self._model.compile(optimizer=opt, loss=self._loss)

    def fit(self, x, y, validation_split=0.2):
        x_2d = np.asarray(x)

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

    def predict(self, x):
        x_2d = np.asarray(x)
        return self._model.predict(x_2d, verbose=0).flatten()