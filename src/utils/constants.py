MAX_WORKERS = 24
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


STATES = ["PR", "SC", "RS"]

TARGETS = ["Residential", "Industrial", "Commercial", "Other"]

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
