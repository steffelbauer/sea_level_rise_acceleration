import os
import sys
import pandas as pd
import numpy as np
import pymc3 as pm
from pymc3.math import switch


if __name__ == "__main__":
    
    # Initial paramters
    method = 'pwlf'  # string for generating name for output files (pwlf - pice-wise linear function)
    niter = 10000  # number of iterations of MCMC process
    station_ids = [7, 9, 20, 22, 23, 24, 25, 32]  # ids of stations that are used in the analysis
    
    # set defaults for command line input arguments [startyear, endyear] if they are not specified
    if len(sys.argv) > 1:
        startyear = str(sys.argv[1])
        print(startyear)
    else:
        startyear = '1921'
    if len(sys.argv) > 2:
        endyear = str(sys.argv[2])
        print(endyear)
    else:
        endyear = '2018'

    # filename and path of the input file (output of unobserved components model) 
    filename = os.path.join('data', 'der', 'ucm', f'level.csv')
    
    # read input file
    all = pd.read_csv(filename, index_col = 0, parse_dates = True)
    all = all[startyear:endyear]  # set data between startyear and endyear
    all = all - all.mean()  # make UCM signal mean free
    all.columns = [str(x) for x in all.columns]  # transform all column names to strings

    # transform all station ids to strings
    station_ids = map(str, station_ids)
    
    # chose only data that belongs to the chosen station_ids
    data = pd.DataFrame(all[station_ids])
    
    # retrieve values and other parameters from data
    values = data.values
    N = data.shape[0]  # number of timesteps
    M = data.columns  # number of stations
    t = np.arange(len(data))  # time vector from 0 to N
    tstart = t.min() # minimum timestep [0]
    tend = t.max()  # maximum timestep [N]

    # add 20 % burn-in phase for MCMC process
    burn_in = int(niter * 0.2)

    # context of MCMC model
    with pm.Model() as model:
        
        # define boundaries for breakpoint
        tau = pm.DiscreteUniform('tau', lower = tstart, upper = tend)

        # define boundaries for slopes (mm/month - multiply with 12 to get mm/year)
        k1 = []
        k2 = []
        for ii, name in enumerate(M):
            k1.append(pm.Uniform(f'k1_{name}', lower = 0.0, upper = 0.5))
            k2.append(pm.Uniform(f'k2_{name}', lower = k1[ii], upper = 0.5))

        # define very conservative boundaries for intecept values (d2 is computed byt other variables, hence, a deterministic variable)
        d1 = []
        d2 = []
        for ii, name in enumerate(M):
            d1.append(pm.Uniform(f'd1_{name}', lower = np.nanmin(values[:, ii]) * 2, upper = np.nanmax(values[:, ii]) * 2))
            d2.append(pm.Deterministic(f'd2_{name}', d1[ii] + (k1[ii] - k2[ii]) * tau))

        # conservative boundaries standard deviation of signal - retireved through empirical trials
        sigma = pm.Uniform("sigma", 2, 200)

        # set model together - > breakpoint switches slopes and intercepts
        d = []
        k = []
        for ii, name in enumerate(M):
            d.append(switch(tau >= t, d1[ii], d2[ii]))
            k.append(switch(tau >= t, k1[ii], k2[ii]))

        # define log-likelihood for measurements under the specific model parameters
        step = pm.NUTS()
        for ii, name in enumerate(M):
            pm.Normal(f'y{name}', mu = k[ii] * t + d[ii], sd = sigma, observed = values[:, ii])

        # MCMC sampling process
        trace = pm.sample(niter, progressbar = True, tune = 4000)

    # Convert trace to dataframe and transform breakpoint from 0 to N units into datetime
    df = pm.backends.tracetab.trace_to_dataframe(trace[burn_in:])
    df['breakpoint'] = [data.index[t] for t in df.tau]
    
    # Save MCMC results to pickle file
    df.to_pickle(os.path.join(f'data/der/breakpoint',  f'breakpoint_{method}_{startyear}_{endyear}.pd'))

    # Print results to console
    slopes = df[df.columns[[x.startswith('k') for x in df.columns]]] * 12
    print(slopes.mean())
    print(df['breakpoint'].mean())


