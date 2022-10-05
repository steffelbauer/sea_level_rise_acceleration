import pandas as pd
import os
import xarray as xr
import numpy as np
import statsmodels
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

def standardize(x: pd.DataFrame) -> pd.DataFrame:
    """Standardize data by subtracting the mean and dividing by the standard deviation

    Args:
        x (np.DataFrame): input dataframe to be standardized

    Returns:
        np.DataFrame: standardized dataframe
    """
    
    return (x - x.mean())/x.std()


if __name__ == '__main__':

    # Input parameters:
    first = '1919-01-01'  # start time
    station_ids = [20, 9, 22, 32, 23, 25, 24, 7]  # ids of chosen stations

    # Read Data:
    # - load the station information
    info = pd.read_csv(os.path.join('data', 'raw', 'rlr_monthly', 'filelist.txt'), 
                       sep = ';', 
                       index_col = 0, 
                       header = None, 
                       names = ('id', 'lat', 'lon', 'name', 'coastline_code', 'station_code', 'quality'))
    info = info.loc[station_ids]

    # - load the station names
    names = pd.read_csv(os.path.join('data', 'raw', 'names.txt'), sep = '\t', index_col = 0, squeeze = True)

    # - nodal cycle data
    filename = os.path.join('data', 'Nodal.nc')
    ds = xr.open_dataset(filename)
    rsl_nodal = ds['rsl_eq']

    # Initialise result dicts
    result = dict()
    coefficients = dict()

    # dataframe for storing the variance inflation factors
    vif_data = pd.DataFrame()
    vif_data["feature"] = ['u_s', 'v_s', 'pres']

    # Loop over stations
    for meas in station_ids:
        
        # print station name
        print("\n\nStation = ", names[meas])
        
        # read station from csv file
        data = pd.read_csv(os.path.join('data', 'der',  'stations', f'{meas}.csv'), index_col=0, parse_dates=[0])
        data.index.freq = 'MS'
        x = data
        x = x.dropna(axis = 0)  # drop nans in measurements
        x = x[first:]
        N = len(x)

        # compute nodal cycle
        nodal_cycle = 18.612958  # nodal cycle period in years
        lat = info.loc[meas]['lat']  # lattitude
        lon = info.loc[meas]['lon']  # longitude
        ampl = -1 * rsl_nodal.interp(x = lat, y = lon).data  # amplitude of nodal cycle
        xdata = x.index.year.astype(int) + x.index.month.astype(float) / 12  # xdata in Julian years
        nodal = ampl * np.cos(2 * np.pi / nodal_cycle * (xdata - 1922.7))  # nodal cycle as cosine curve
        nodal = pd.Series(nodal, index = x.index)  # transform nodal cycle to pandas series

        # standardize exogenous variables
        x[['u_s', 'v_s', 'pres']] = x[['u_s', 'v_s', 'pres']].apply(standardize, axis = 0)
        X = x[['u_s', 'v_s', 'pres']]
        
        # compute variance inflation factors for all exogenous variables
        vif_data[f"VIF_{meas}"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
        print(vif_data)

        # initialise keyword arguments for UCM
        ucm_kwargs = dict(exog = x[['u_s', 'v_s', 'pres']], 
                          seasonal = 12, 
                          stochastic_seasonal = False, 
                          level = 'llevel', 
                          use_exact_diffuse = False)

        # make signal mean free by subtracting the mean of the sea-level:
        x['height'] = x['height'] - x['height'].mean()

        # different options:
        x['height'] = x['height'] + nodal
    
        # fit unobserved components model (UCM)
        model = statsmodels.tsa.statespace.structural.UnobservedComponents(x['height'], **ucm_kwargs)
        res = model.fit(disp = 0)
        print(res.summary())

        # Output nodal amplitudes
        comp_ampl = ampl
        
        # Extract level from UCM
        level = pd.Series(index = x.index, data = res.level.smoothed, name = meas)
        result[meas] = level
        level = level - level.mean()

    # Save results to pandas pickle and csv file:
    result = pd.DataFrame(result)
    # result.to_pickle(os.path.join('data', 'der', 'ucm', f'level_{method}.pd'))
    # result.to_csv(os.path.join('data', 'der', 'ucm', f'level_{method}.csv'))
    result.to_csv(os.path.join('data', 'der', 'ucm', f'level.csv'))

    (pd.DataFrame([result[col]+150*i for i, col in enumerate(result.columns)]).T).plot()
    plt.show()