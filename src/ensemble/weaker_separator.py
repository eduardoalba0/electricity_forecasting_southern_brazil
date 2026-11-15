import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from wrappers.tensorflow_wrappers import LSTM_Wrapper, MLP_Wrapper

class WSBagging:
    def __init__(self,
                 strong_predictor=None,
                 weak_predictors=None,
                 strong_weight=0.5,
                 weak_weight=0.5,
                 bootstrap=True,
                 random_patches=True,
                 bootstrap_ratio=0.5,
                 patches_ratio=0.5,
                 early_stopping=5,
                 random_state=42):

        if not (0.0 < bootstrap_ratio <= 1.0) or not (0.0 < patches_ratio <= 1.0):
            raise ValueError("Ratios must be in (0.0, 1.0] interval.")


        self.strong_predictor = strong_predictor
        self.weak_predictors = weak_predictors if weak_predictors is not None else []
        self.strong_weight = strong_weight
        self.weak_weight = weak_weight

        self.bootstrap = bootstrap
        self.random_patches = random_patches
        self.bootstrap_ratio = bootstrap_ratio
        self.patches_ratio = patches_ratio
        self.early_stopping = early_stopping

        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)
        self.feature_index_ = []

    def fit(self, x, y, validation_split=0.2):
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)

        if validation_split > 0:
            x_train, x_val, y_train, y_val = train_test_split(x_arr, y_arr,
                                                              test_size=validation_split,
                                                              shuffle=False)
        else:
            x_train, y_train = x_arr, y_arr
            x_val, y_val = None, None

        n_samples, n_features = x_train.shape
        sample_size = max(1, int(n_samples * self.bootstrap_ratio))
        patch_size = max(1, int(n_features * self.patches_ratio))

        if self.strong_predictor is not None:
            self.strong_predictor.fit(x_train, y_train)

        candidates = list(self.weak_predictors)
        self.weak_predictors = []
        self.feature_index_ = []

        best_mse = float('inf')
        patience_counter = 0
        best_state = ([], [])

        if validation_split > 0:
            baseline_preds = self.predict(x_val)
            best_mse = mean_squared_error(y_val, baseline_preds)

        for i, predictor in enumerate(candidates):
            if self.bootstrap:
                start = int(self.rng.integers(0, n_samples - sample_size + 1))
                row_idx = np.arange(start, start + sample_size)
            else:
                row_idx = np.arange(n_samples)

            if self.random_patches:
                col_idx = self.rng.choice(n_features, size=patch_size, replace=False)
            else:
                col_idx = np.arange(n_features)

            predictor.fit(x_train[np.ix_(row_idx, col_idx)], y_train[row_idx])
            self.weak_predictors.append(predictor)
            self.feature_index_.append(col_idx)

            if validation_split > 0:
                current_preds = self.predict(x_val)
                current_mse = mean_squared_error(y_val, current_preds)

                if current_mse < best_mse:
                    best_mse = current_mse
                    patience_counter = 0
                    best_state = (list(self.weak_predictors), list(self.feature_index_))
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping:
                        break

        if validation_split > 0 and len(best_state[0]) > 0:
            self.weak_predictors, self.feature_index_ = best_state

        return self

    def predict(self, x):
        x_arr = np.asarray(x)
        n_preds = x_arr.shape[0]
        strong_preds = []

        if self.strong_predictor is not None:
            strong_preds = np.asarray(self.strong_predictor.predict(x_arr)).reshape(-1)

        if len(self.weak_predictors) == 0:
            return np.asarray(strong_preds) if self.strong_predictor is not None else np.zeros(n_preds)

        all_weak_preds = []
        for j in range(len(self.weak_predictors)):
            predictor = self.weak_predictors[j]
            col_idx = self.feature_index_[j]

            x_subset = x_arr[:, col_idx]
            pred_j = np.asarray(predictor.predict(x_subset)).reshape(-1)
            all_weak_preds.append(pred_j)

        weight_sum = self.strong_weight + self.weak_weight

        final_preds = []
        for i in range(n_preds):
            weak_preds_i = []
            for j in range(len(self.weak_predictors)):
                weak_preds_i.append(float(all_weak_preds[j][i]))
            weak_preds_i = np.array(weak_preds_i)

            mean_preds = np.mean(weak_preds_i)
            std_preds = np.std(weak_preds_i)

            if std_preds == 0:
                weak_mean = float(mean_preds)
            else:
                z_score = (weak_preds_i - mean_preds) / std_preds
                valid_weak_preds = weak_preds_i[np.abs(z_score) < 1.0]

                if len(valid_weak_preds) == 0:
                    weak_mean = float(mean_preds)
                else:
                    weak_mean = float(np.mean(valid_weak_preds))

            if self.strong_predictor is not None:
                strong_pred_i = float(strong_preds[i])
                pred = ((strong_pred_i * self.strong_weight) + (weak_mean * self.weak_weight)) / weight_sum
            else:
                pred = weak_mean

            final_preds.append(pred)

        return np.array(final_preds)

class MLP(WSBagging):

    def __init__(self,
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

        self.mlp_strong_hidden = mlp_strong_hidden
        self.mlp_strong_epochs = mlp_strong_epochs
        self.mlp_strong_batch_size = mlp_strong_batch_size
        self.mlp_strong_activation = mlp_strong_activation

        self.n_weak_mlps = max_mlps
        self.max_mlp_hidden = max_mlp_hidden
        self.max_mlp_epochs = max_mlp_epochs
        self.max_mlp_batch_size = max_mlp_batch_size
        self.build_model()

    def build_model(self):
        activations = ["linear", "mish", "sigmoid", "softmax", "softplus", "softsign", "tanh", 'elu', 'selu']
        self.weak_predictors = []

        self.strong_predictor = MLP_Wrapper(
                hidden_layers=self.mlp_strong_hidden,
                epochs=self.mlp_strong_epochs,
                batch_size=self.mlp_strong_batch_size,
                hidden_activation=self.mlp_strong_activation,
                early_stopping=self.early_stopping
        )
        for _ in range(self.n_weak_mlps):
            mlp = MLP_Wrapper(
                hidden_layers=int(self.rng.integers(1, self.max_mlp_hidden, endpoint=True)),
                epochs=int(self.rng.integers(1, self.max_mlp_epochs, endpoint=True)),
                batch_size=int(self.rng.integers(1, self.max_mlp_batch_size, endpoint=True)),
                hidden_activation=self.rng.choice(activations),
                early_stopping=self.early_stopping
            )
            self.weak_predictors.append(mlp)

class LSTMMLP(WSBagging):
    def __init__(self,
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

        self.lstm_units = lstm_units
        self.lstm_epochs = lstm_epochs
        self.lstm_batch_size = lstm_batch_size
        self.lstm_activation = lstm_activation

        self.n_weak_mlps = max_mlps
        self.max_mlp_hidden = max_mlp_hidden
        self.max_mlp_epochs = max_mlp_epochs
        self.max_mlp_batch_size = max_mlp_batch_size
        self.build_model()

    def build_model(self):
        activations = ["linear", "mish", "sigmoid", "softmax", "softplus", "softsign", "tanh", 'elu', 'selu']
        self.weak_predictors = []

        self.strong_predictor = LSTM_Wrapper(
            units=self.lstm_units,
            epochs=self.lstm_epochs,
            lstm_activation=self.lstm_activation,
            batch_size=self.lstm_batch_size,
            early_stopping=self.early_stopping
        )

        for _ in range(self.n_weak_mlps):
            mlp = MLP_Wrapper(
                hidden_layers=int(self.rng.integers(1, self.max_mlp_hidden, endpoint=True)),
                epochs=int(self.rng.integers(1, self.max_mlp_epochs, endpoint=True)),
                batch_size=int(self.rng.integers(1, self.max_mlp_batch_size, endpoint=True)),
                hidden_activation=self.rng.choice(activations),
                early_stopping=self.early_stopping
            )
            self.weak_predictors.append(mlp)