import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.callbacks import EarlyStopping

class LSTM_Wrapper:
    def __init__(self,
                 units,
                 epochs,
                 batch_size,
                 lstm_activation='relu',
                 output_activation='linear',
                 optimizer='adam',
                 loss='mse',
                 early_stopping=5,
                 validation_split=0.2):
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.lstm_activation = lstm_activation
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.loss = loss
        self.early_stopping = early_stopping
        self.validation_split = validation_split
        self.model = None


    def fit(self, x, y):
        x_3d = np.asarray(x).reshape((x.shape[0], x.shape[1], 1))

        self.model = Sequential([
            Input(shape=(x_3d.shape[1], 1)),
            LSTM(self.units, activation=self.lstm_activation),
            Dense(1, activation=self.output_activation)
        ])
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

        callbacks = []
        if self.early_stopping > 0:
            callbacks.append(EarlyStopping(
                monitor='val_loss', patience=self.early_stopping, restore_best_weights=True
            ))

        self.model.fit(
            x_3d, y, epochs=self.epochs, batch_size=self.batch_size,
            validation_split=self.validation_split, callbacks=callbacks, verbose=0
        )
        return self

    def predict(self, x):
        x_3d = np.asarray(x).reshape((x.shape[0], x.shape[1], 1))
        return self.model.predict(x_3d, verbose=0).flatten()


class MLP_Wrapper:
    def __init__(self,
                 hidden_layers,
                 epochs,
                 batch_size,
                 hidden_activation='relu',
                 output_activation='linear',
                 optimizer='adam',
                 loss='mse',
                 early_stopping=5,
                 validation_split=0.2):

        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.loss = loss
        self.early_stopping = early_stopping
        self.validation_split = validation_split
        self.model = None

    def fit(self, x, y):
        self.model = Sequential()
        self.model.add(Input(shape=(x.shape[1],)))

        self.model.add(Dense(self.hidden_layers, activation=self.hidden_activation))

        self.model.add(Dense(1, activation=self.output_activation))
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

        callbacks = []
        if self.early_stopping > 0:
            callbacks.append(EarlyStopping(
                monitor='val_loss', patience=self.early_stopping, restore_best_weights=True
            ))

        self.model.fit(
            x, y, epochs=self.epochs, batch_size=self.batch_size,
            validation_split=self.validation_split, callbacks=callbacks, verbose=0
        )
        return self

    def predict(self, x):
        return self.model.predict(x, verbose=0).flatten()


