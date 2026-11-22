import os
from pathlib import Path



BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DATA_DIR = Path(BASE_DIR) / "data" / "processed"
RESULTS_DIR = Path(BASE_DIR) / "paper" / "results"
HYPERPARAMS_DIR = RESULTS_DIR / "hyperparameters"
FEATURES_DIR = RESULTS_DIR / "features"
SHAP_DIR = RESULTS_DIR / "shap"
PERFORMANCE_DIR = RESULTS_DIR / "performance"

MAX_WORKERS = 20
N_JOBS = 5
URL_ADS = "https://ads.atmosphere.copernicus.eu/api"
URL_CDS = "https://cds.climate.copernicus.eu/api"

CAMS_VARS = [
    "10 metre U wind component",
    "10 metre V wind component",
    "2 metre dewpoint temperature",
    "2 metre temperature",
]

ERA5_VARS = [
    "Mean sea level pressure",
    "Surface pressure",
    "Total precipitation",
    "Total cloud cover",
]

WEATHER_AGG_STATS = ["mean", "max", "min", "sum", "std"]

STATES = ["PR", "SC", "RS"]

TARGETS = ["Residential", "Industrial", "Commercial", "Other"]

CONSUMERS_VARS = ["Residential Consumers", "Industrial Consumers", "Commercial Consumers", "Other Consumers"]

CONSUMERS_SLUG = ["residential", "commercial", "industrial", "other"]

REQUIRED_VARS = [
    "State_PR",
    "State_RS",
    "State_SC",
    "Month_1",
    "Month_2",
    "Month_3",
    "Month_4",
    "Month_5",
    "Month_6",
    "Month_7",
    "Month_8",
    "Month_9",
    "Month_10",
    "Month_11",
    "Month_12"
]

QUADRANTS = ["Q1", "Q2", "Q3", "Q4"]

HOURS = ["0900", "1800", "2100"]

BATCH_SIZES = [8, 16, 32, 64, 128, 256, 512]

ACTIVATIONS = [
    "linear",
    "mish",
    "sigmoid",
    "softmax",
    "softplus",
    "softsign",
    "tanh",
    "elu",
    "selu"]


MLP_BOUNDS = [
    (10, 200),                  # hidden_layers (units)
    (20, 100),                  # epochs
    (0, len(BATCH_SIZES) - 1),  # batch_size
    (0, len(ACTIVATIONS) - 1),  # activation index in ACTIVATIONS
    (0.0005, 0.05),             # learning_rate
]

LSTM_BOUNDS = [
    (10, 200),                  # units
    (20, 100),                  # epochs
    (0, len(BATCH_SIZES) - 1),  # batch_size
    (0, len(ACTIVATIONS) - 1),  # activation index in ACTIVATIONS
    (0.0005, 0.05),             # learning_rate
]

LSTMMLPWSB_BOUNDS = [
    (10, 200),                  # lstm_units
    (0, len(ACTIVATIONS) - 1),  # lstm activation index in ACTIVATIONS
    (2, 20),                    # n_weak_mlps
    (10, 50) ,                  # max_mlp_hidden
    (20, 100),                  # epochs
    (0, len(BATCH_SIZES) - 1),  # batch_size
    (0.0005, 0.05),             # lstm_learning_rate
    (0.0005, 0.05),             # mlp_learning_rate
    (0.05, 0.90),               # strong_weight
    (0.2, 1)                    # patches_ratio
]

MLPWSB_BOUNDS = [
    (10, 200),                  # mlp_strong_hidden (units)
    (0, len(ACTIVATIONS) - 1),  # mlp_strong_activation index in ACTIVATIONS
    (2, 20),                    # n_weak_mlps
    (10, 50),                   # max_mlp_hidden
    (20, 100),                  # epochs
    (0, len(BATCH_SIZES) - 1),  # batch_size
    (0.0005, 0.05),             # mlp_strong_learning_rate
    (0.0005, 0.05),             # mlp_weak_learning_rate
    (0.05, 0.90),               # strong_weight
    (0.2, 1)                    # patches_ratio
]

TensorLSTMMLPWSB_BOUNDS = [
    (10, 200),                  # lstm_units
    (0, len(ACTIVATIONS) - 1),  # lstm activation index in ACTIVATIONS
    (2, 20),                    # n_weak_mlps
    (10, 50) ,                  # max_mlp_hidden
    (20, 100),                  # epochs
    (20, 100),                  # epochs
    (0.0005, 0.05),             # learning_rate
    (0.05, 0.90),               # strong_weight
    (0.2, 1)                    # patches_ratio
]

TensorMLPWSB_BOUNDS = [
    (10, 200),                  # mlp_strong_hidden (units)
    (0, len(ACTIVATIONS) - 1),  # mlp_strong_activation index in ACTIVATIONS
    (2, 20),                    # n_weak_mlps
    (10, 50),                   # max_mlp_hidden
    (20, 100),                  # epochs
    (0, len(BATCH_SIZES) - 1),  # batch_size
    (0.0005, 0.05),             # learning_rate
    (0.05, 0.90),               # strong_weight
    (0.2, 1)                    # patches_ratio
]


def get_models():
    from ensemble import MLPWSB, LSTMMLPWSB, TensorMLPWSB, TensorLSTMMLPWSB
    from wrappers import MLPWrapper, LSTMWrapper

    return {
        "mlp": ("MLP", MLPWrapper, list(MLP_BOUNDS)),
        "lstm": ("LSTM", LSTMWrapper, list(LSTM_BOUNDS)),
        "mlp-wsb": ("MLP-WSB", MLPWSB, list(MLPWSB_BOUNDS)),
        "lstm-mlp-wsb": ("LSTM-MLP-WSB", LSTMMLPWSB, list(LSTMMLPWSB_BOUNDS)),
        "mlp-wsb-tf": ("MLP-WSB-TF", TensorMLPWSB, list(TensorMLPWSB_BOUNDS)),
        "lstm-mlp-wsb-tf": ("LSTM-MLP-WSB-TF", TensorLSTMMLPWSB, list(TensorLSTMMLPWSB_BOUNDS)),
    }

OPTIMIZERS = ["COA", "PSO"]

SEEDS = range(1000, 20000+1, 1000)

FEATURE_THRESHOLD_BOUND = (0.0, 1.0)
