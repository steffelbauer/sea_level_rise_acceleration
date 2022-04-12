import os
import numpy as np
import pandas as pd
import xarryay as xr
from typing import Union

# set datapath variable to folder where the raw data is stored
datapath = os.path.join('data', 'raw')

# download current wind (u and v direction) and pressure data from 1948 onwards from PSL Gridded Datasets (https://psl.noaa.gov/repository/a/psdgrids)
ds_u_present = xr.open_dataset(os.path.join(datapath, 'present', 'uwnd.10m.mon.mean.nc'))
ds_v_present = xr.open_dataset(os.path.join(datapath, 'present', 'vwnd.10m.mon.mean.nc'))
ds_p_present = xr.open_dataset(os.path.join(datapath, 'present', 'pres.sfc.mon.mean.nc'))

# download past wind (u and v direction) and pressure data until 1948  from 20th Century Reanalysis (V3)  (https://psl.noaa.gov/data/gridded/data.20thC_ReanV3.html)
ds_u_past = xr.open_dataset(os.path.join(datapath, 'past', 'uwnd.10m.mon.mean.nc'))['uwnd']
ds_v_past = xr.open_dataset(os.path.join(datapath, 'past', 'vwnd.10m.mon.mean.nc'))['vwnd']
ds_p_past = xr.open_dataset(os.path.join(datapath, 'past', 'pres.sfc.mon.mean_v3.nc'))['pres']


def combine_series(present: pd.DataFrame, past: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
    """Combine two pandas dataframes or series into one by appending one to the other

    Args:
        present (pd.DataFrame): dataframe containing data from the PSL Gridded Dataset
        past (pd.DataFrame): dataframe containing data from the 20th Century Reanalysis (V3) project

    Returns:
        Union[pd.DataFrame, pd.Series]: comnined dataframe 
    """
    inter = present.index.intersection(past.index)
    result = pd.concat([past[:inter[0]], present[inter[1]:]])
    
    return result


if __name__ == '__main__':

    # get information of different stations from the rlr dataset
    info = pd.read_csv(os.path.join(datapath, 'rlr_monthly', 'filelist.txt'), 
                       sep = ';', 
                       index_col = 0, 
                       header = None, 
                       names = ('id', 'lat', 'lon', 'name', 'coastline_code', 'station_code', 'quality'))
    
    # chose station ids corresponding to the used tide gauge stations
    station_ids = [7, 9, 20, 22, 23, 24, 25, 32]

    info = info.loc[station_ids]

    df = []
    for station in station_ids:
        print(station)
        name = info.loc[station]['name'].strip().split(' ')[0]  # name of station
        lat = info.loc[station]['lat']  # latitude
        lon = info.loc[station]['lon']  # longitude
        
        # if lat and lon are not floats but list of floats, only chose first element
        if not isinstance(lat, np.float64):
            lat = lat.iloc[0]
        if not isinstance(lon, np.float64):
            lon = lon.iloc[0]

        # read tide gauge measurements
        s = pd.read_csv(os.path.join(datapath, 'rlr_monthly', 'data', f'{station}.rlrdata'), header = None, index_col = 0, sep = ';', 
                        usecols = [0, 1], names = ('julian', 'height'))
        
        # values smaller than 0 are ignored (values in rlr data is set to positive values)
        s[s < 0] = np.nan
        s = s.squeeze()
        s.index = pd.date_range(f'{int(s.index[0])}-01-01', freq = 'MS', periods = len(s))

        # read pressures and wind data from the reanalysis (past) and the grid dataset (present)
        p1 = ds_p_present.interp(lat = lat, lon = lon).to_dataframe()['pres']
        p2 = ds_p_past.interp(lat = lat, lon = lon).to_dataframe()['pres']

        u1 = ds_u_present.interp(lat = lat, lon = lon).to_dataframe()['uwnd']
        u2 = ds_u_past.interp(lat = lat, lon = lon).to_dataframe()['uwnd']

        v1 = ds_v_present.interp(lat = lat, lon = lon).to_dataframe()['vwnd']
        v2 = ds_v_past.interp(lat = lat, lon = lon).to_dataframe()['vwnd']

        # combine data past and present datasets
        p = combine_series(p1, p2)
        u = combine_series(u1, u2)
        v = combine_series(v1, v2)

        # squared wind but keep sign
        u_square = u ** 2 * np.sign(u)
        v_square = v ** 2 * np.sign(v)

        u_square.name = 'u2'
        v_square.name = 'v2'

        # compute wind stresses according to 
        u_stress = u * np.sqrt(u**2 + v**2)
        v_stress = v * np.sqrt(u**2 + v**2)

        u_stress.name = 'u_s'
        v_stress.name = 'v_s'

        # generate dataframe for single station containing sea-level measurement, pressure and wind and save to csv and pickle file
        data = pd.concat([s, p, u_square, v_square, u_stress, v_stress], axis = 1)
        data.to_pickle(os.path.join('data', 'der', 'stations', f'{station}.pd'))
        data.to_csv(os.path.join('data', 'der', 'stations', f'{station}.csv'))

        s.name = name  # station name is the name of the series
        df.append(s)  # append series to dataframe containing all sea-level measurements of the stations

    # save dataframe with all measurements
    df = pd.concat(df, axis = 1)
    df.to_pickle(os.path.join('data', 'der', 'stations', 'all_stations_psmsl.pd'))
    df.to_csv(os.path.join('data', 'der', 'stations', 'all_stations_psmsl.csv'))
