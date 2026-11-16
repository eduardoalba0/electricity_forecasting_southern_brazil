import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class WSBagging:
    def __init__(self,
                 inputs,
                 timesteps,
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

        self._inputs = inputs
        self._strong_predictor = None
        self._weak_predictors =  []
        self._strong_weight = strong_weight
        self._weak_weight = weak_weight

        self._bootstrap = bootstrap
        self._random_patches = random_patches
        self._bootstrap_ratio = bootstrap_ratio
        self._patches_ratio = patches_ratio
        self._early_stopping = early_stopping
        self._timesteps = timesteps

        self._random_state = random_state
        self._rng = np.random.default_rng(self._random_state)
        self._feature_index = []

    def _build_model(self):
        raise NotImplementedError

    def fit(self, x, y, validation_split=0.2):
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)

        if validation_split > 0:
            x_train, x_val, y_train, y_val = train_test_split(x_arr, y_arr, test_size=validation_split, shuffle=False)
        else:
            x_train, y_train = x_arr, y_arr
            x_val, y_val = None, None

        if self._strong_predictor is not None:
            self._strong_predictor.fit(x_train, y_train)

        n_samples = x_train.shape[0]
        sample_size = max(1, int(n_samples * self._bootstrap_ratio))

        if sample_size < self._timesteps:
            sample_size = self._timesteps

        patch_size = max(1, int(self._inputs * self._patches_ratio))

        candidates = list(self._weak_predictors)
        self._weak_predictors = []
        self._feature_index = []

        best_mse = float('inf')
        patience_counter = 0
        best_state = ([], [])

        if validation_split > 0:
            baseline_preds = self.predict(x_val)
            best_mse = mean_squared_error(y_val, baseline_preds)

        for i, predictor in enumerate(candidates):
            if self._bootstrap:
                start = int(self._rng.integers(0, n_samples - sample_size + 1))
                row_idx = np.arange(start, start + sample_size)
            else:
                row_idx = np.arange(n_samples)

            if self._random_patches:
                col_idx = self._rng.choice(self._inputs, size=patch_size, replace=False)
            else:
                col_idx = np.arange(self._inputs)

            predictor.fit(x_train[np.ix_(row_idx, col_idx)], y_train[row_idx])
            self._weak_predictors.append(predictor)
            self._feature_index.append(col_idx)

            if validation_split > 0:
                current_preds = self.predict(x_val)
                current_mse = mean_squared_error(y_val, current_preds)

                if current_mse < best_mse:
                    best_mse = current_mse
                    patience_counter = 0
                    best_state = (list(self._weak_predictors), list(self._feature_index))
                else:
                    patience_counter += 1
                    if patience_counter >= self._early_stopping:
                        break

        if validation_split > 0 and len(best_state[0]) > 0:
            self._weak_predictors, self._feature_index = best_state

        return self

    def predict(self, x):
        x_arr = np.asarray(x)
        strong_preds = []

        if self._strong_predictor is not None:
            strong_preds = np.asarray(self._strong_predictor.predict(x_arr)).reshape(-1)

        n_preds = len(x_arr)

        if len(self._weak_predictors) == 0:
            return np.asarray(strong_preds) if self._strong_predictor is not None else np.zeros(n_preds)

        all_weak_preds = []
        for j in range(len(self._weak_predictors)):
            predictor = self._weak_predictors[j]
            col_idx = self._feature_index[j]

            x_subset = x_arr[:, col_idx]
            pred_j = np.asarray(predictor.predict(x_subset)).reshape(-1)
            all_weak_preds.append(pred_j)

        weight_sum = self._strong_weight + self._weak_weight

        final_preds = []
        for i in range(n_preds):
            weak_preds_i = []
            for j in range(len(self._weak_predictors)):
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

            if self._strong_predictor is not None:
                strong_pred_i = float(strong_preds[i])
                pred = ((strong_pred_i * self._strong_weight) + (weak_mean * self._weak_weight)) / weight_sum
            else:
                pred = weak_mean

            final_preds.append(pred)

        return np.array(final_preds)
