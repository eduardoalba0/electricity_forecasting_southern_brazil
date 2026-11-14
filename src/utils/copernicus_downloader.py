import cdsapi
import os
import pandas as pd
import sys
from utils.grib_to_df import grib_to_df

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)
from credentials import KEY_ADS_CDS

URL_ADS = "https://ads.atmosphere.copernicus.eu/api"
URL_CDS = "https://cds.climate.copernicus.eu/api"

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PATH = os.path.join(BASE_DIR, "data", "raw")

os.makedirs(os.path.join(PATH, "cams"), exist_ok=True)
os.makedirs(os.path.join(PATH, "era5"), exist_ok=True)

regions = {
    "PR": {
        "Q1": [-22.71, -54.34, -24.50, -51.20],
        "Q2": [-22.89, -51.20, -24.50, -49.16],
        "Q3": [-24.50, -54.34, -26.28, -51.20],
        "Q4": [-24.50, -51.20, -26.63, -48.10]
    },
    "SC": {
        "Q1": [-26.10, -53.80, -26.85, -51.00],
        "Q2": [-26.10, -51.00, -27.50, -48.30],
        "Q3": [-26.85, -53.80, -27.70, -51.00],
        "Q4": [-27.50, -51.00, -29.20, -48.30]
    },
    "RS": {
        "Q1": [-27.21, -56.56, -30.07, -53.02],
        "Q2": [-27.70, -53.00, -30.07, -49.76],
        "Q3": [-30.07, -56.64, -32.06, -53.00],
        "Q4": [-30.07, -53.00, -32.22, -50.93]
    }
}
def process_cams_request(state, region_name, region_coords):
    filename_grib = os.path.join(PATH, "cams", f"{state}_{region_name}.grib")
    filename_csv = os.path.join(PATH, "cams", f"{state}_{region_name}.csv")

    if os.path.exists(filename_csv):
        print(f"File already exists: {filename_csv}")
        return

    request = {
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_dewpoint_temperature",
            "2m_temperature"
        ],
        "date": "2004-01-01/2025-12-31",
        "time": ["09:00", "18:00", "21:00"], # <-- UTC // LOCAL: 06:00, 15:00, 18:00
        "format": "grib",
        "area": region_coords
    }


    try:
        if not os.path.exists(filename_grib):
            print(f"Attempting to download {filename_grib}")
            dataset = "cams-global-reanalysis-eac4"
            client_ADS = cdsapi.Client(url=URL_ADS, key=KEY_ADS_CDS)
            client_ADS.retrieve(dataset, request, filename_grib)

        print(f"Starting conversion: {state}_{region_name}")

        lista_dfs = grib_to_df(filename_grib)
        df_final = pd.concat(lista_dfs, ignore_index=True)
        df_final.sort_values(by=['time', 'latitude', 'longitude'], inplace=True)
        df_final.to_csv(filename_csv, index=False, decimal='.', sep=",")

        print(f"Successfully finished CAMS {state} / {region_name}")

    except Exception as e:
        print(f"Error processing {state} / {region_name}: {e}")

def process_era5_region(state, region_name, region_coords):
    filename_grib = os.path.join(PATH, "era5", f"{state}_{region_name}.grib")
    filename_csv = os.path.join(PATH, "era5", f"{state}_{region_name}.csv")

    if os.path.exists(filename_csv):
        print(f"File already exists: {filename_csv}")
        return

    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "mean_sea_level_pressure",
            "surface_pressure",
            "total_precipitation",
            "total_cloud_cover"
        ],
        "year": [str(y) for y in range(2004, 2026)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": ["09:00", "18:00", "21:00"],  # <-- UTC // LOCAL: 06:00, 15:00, 18:00
        "download_format": "unarchived",
        "area": region_coords
    }


    try:
        if not os.path.exists(filename_grib):
            print(f"Attempting to download {filename_grib}")
            dataset = "reanalysis-era5-single-levels"
            client_CDS = cdsapi.Client(url=URL_CDS, key=KEY_ADS_CDS)
            client_CDS.retrieve(dataset, request, filename_grib)

        print(f"Starting conversion: {state}_{region_name}")

        lista_dfs = grib_to_df(filename_grib)
        df_final = pd.concat(lista_dfs, ignore_index=True)
        df_final.sort_values(by=['time', 'latitude', 'longitude'], inplace=True)
        df_final.to_csv(filename_csv, index=False, decimal='.', sep=",")

        print(f"Successfully finished ERA5 {state} / {region_name}")

    except Exception as e:
        print(f"Error processing {state} / {region_name}: {e}")
