import pygrib
import pandas as pd
import numpy as np
from collections import defaultdict

def grib_to_df(filename_grib):
    grbs = pygrib.open(filename_grib)

    grb = grbs.read(1)[0]
    lats, lons = grb.latlons()
    flat_lats = np.round(lats.flatten(), 4)
    flat_lons = np.round(lons.flatten(), 4)

    grbs.seek(0)

    data_by_time = defaultdict(dict)

    for grb in grbs:
        dt = grb.validDate
        var_name = grb.name
        data_by_time[dt][var_name] = grb.values.flatten()

    grbs.close()

    registros = []
    for dt, vars_dict in data_by_time.items():
        df_dt = pd.DataFrame({
            'time': dt,
            'latitude': flat_lats,
            'longitude': flat_lons,
            **vars_dict
        })
        registros.append(df_dt)

    return registros