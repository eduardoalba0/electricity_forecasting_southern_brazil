import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class WSBagging:
    def __init__(self,
                 inputs,
                 timesteps,
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

        self._inputs = inputs
        self._strong_predictor = strong_predictor
        self._weak_predictors = weak_predictors if weak_predictors is not None else []
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
        self._feature_index_ = []

    def _align_weak_data(self, x, y=None):
        if y is not None:
            return x, y
        return x

    def _align_targets(self, y):
        return y

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

        x_weak, y_weak = self._align_weak_data(x_train, y_train)

        n_samples_weak, n_features = x_weak.shape
        sample_size = max(1, int(n_samples_weak * self._bootstrap_ratio))

        if sample_size < self._timesteps:
            sample_size = self._timesteps

        patch_size = max(1, int(self._inputs * self._patches_ratio))

        candidates = list(self._weak_predictors)
        self._weak_predictors = []
        self._feature_index_ = []

        best_mse = float('inf')
        patience_counter = 0
        best_state = ([], [])

        if validation_split > 0:
            baseline_preds = self.predict(x_val)
            y_val_adj = self._align_targets(y_val)
            best_mse = mean_squared_error(y_val_adj, baseline_preds)

        for i, predictor in enumerate(candidates):
            if self._bootstrap:
                start = int(self._rng.integers(0, n_samples_weak - sample_size + 1))
                row_idx = np.arange(start, start + sample_size)
            else:
                row_idx = np.arange(n_samples_weak)

            if self._random_patches:
                col_idx = self._rng.choice(n_features, size=patch_size, replace=False)
            else:
                col_idx = np.arange(n_features)

            predictor.fit(x_weak[np.ix_(row_idx, col_idx)], y_weak[row_idx])
            self._weak_predictors.append(predictor)
            self._feature_index_.append(col_idx)

            if validation_split > 0:
                current_preds = self.predict(x_val)
                current_mse = mean_squared_error(y_val_adj, current_preds)

                if current_mse < best_mse:
                    best_mse = current_mse
                    patience_counter = 0
                    best_state = (list(self._weak_predictors), list(self._feature_index_))
                else:
                    patience_counter += 1
                    if patience_counter >= self._early_stopping:
                        break

        if validation_split > 0 and len(best_state[0]) > 0:
            self._weak_predictors, self._feature_index_ = best_state

        return self

    def predict(self, x):
        x_arr = np.asarray(x)
        strong_preds = []

        if self._strong_predictor is not None:
            strong_preds = np.asarray(self._strong_predictor.predict(x_arr)).reshape(-1)

        x_weak = self._align_weak_data(x_arr)
        n_preds = len(x_weak)

        if len(self._weak_predictors) == 0:
            return np.asarray(strong_preds) if self._strong_predictor is not None else np.zeros(n_preds)

        all_weak_preds = []
        for j in range(len(self._weak_predictors)):
            predictor = self._weak_predictors[j]
            col_idx = self._feature_index_[j]

            x_subset = x_weak[:, col_idx]
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
