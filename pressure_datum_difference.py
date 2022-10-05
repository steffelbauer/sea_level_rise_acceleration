import xarray as xr
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# This script shows the shift in the pressure datums between the two reanalysis datasets that are used.
# The offset/discontinuity is corrected in the prepare_data.py script.


# set datapath variable to folder where the raw data is stored
datapath = os.path.join('data', 'raw')

ds_u_present = xr.open_dataset(os.path.join(datapath, 'present', 'uwnd.10m.mon.mean.nc'))
ds_v_present = xr.open_dataset(os.path.join(datapath, 'present', 'vwnd.10m.mon.mean.nc'))
ds_p_present = xr.open_dataset(os.path.join(datapath, 'present', 'pres.sfc.mon.mean.nc'))

ds_u_past = xr.open_dataset(os.path.join(datapath, 'past', 'uwnd.10m.mon.mean.nc'))['uwnd']
ds_v_past = xr.open_dataset(os.path.join(datapath, 'past', 'vwnd.10m.mon.mean.nc'))['vwnd']
ds_p_past = xr.open_dataset(os.path.join(datapath, 'past', 'pres.sfc.mon.mean_v3.nc'))['pres']


# get information of different stations from the rlr dataset
info = pd.read_csv(os.path.join(datapath, 'rlr_monthly', 'filelist.txt'), 
                    sep = ';', 
                    index_col = 0, 
                    header = None, 
                    names = ('id', 'lat', 'lon', 'name', 'coastline_code', 'station_code', 'quality'))

# chose station ids corresponding to the used tide gauge stations
station_ids = [7, 9, 20, 22, 23, 24, 25, 32]
station = 25

name = info.loc[station]['name'].strip().split(' ')[0]  # name of station
lat = info.loc[station]['lat']  # latitude
lon = info.loc[station]['lon']  # longitude

# if lat and lon are not floats but list of floats, only chose first element
if not isinstance(lat, np.float64):
    lat = lat.iloc[0]
if not isinstance(lon, np.float64):
    lon = lon.iloc[0]

p1 = ds_p_present.interp(lat = lat, lon = lon).to_dataframe()['pres']
p2 = ds_p_past.interp(lat = lat, lon = lon).to_dataframe()['pres']


fig, ax = plt.subplots()

p2.plot(label='NOAA/CIRES/DOE 20th Century Reanalysis V3')
p1.plot(label='NCEP-NCAR Reanalysis 1')

plt.title(f'Pressure at station {name}', fontsize=16)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Pressure (Pa)', fontsize=14)
plt.legend()
plt.show()



