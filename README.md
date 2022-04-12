# Evidence of Regional Sea-Level Rise Acceleration for the North Sea

The scripts in this repository were used to produce the results presented in the draft paper submitted to Environmental Research Letters called "Evidence of Regional Sea-Level Rise Acceleration for the North Sea".

To run the code, please follow the following steps:

Download following data files from those repositories:
* Wind (u and v direction) and pressure data from 1948 onwards from PSL Gridded Datasets (https://psl.noaa.gov/repository/a/psdgrids)
* Past wind (u and v direction) and pressure data until 1948  from 20th Century Reanalysis (V3)  (https://psl.noaa.gov/data/gridded/data.20thC_ReanV3.html)

Then execute the code in following order:
* prepare_data.py: script that puts the data together from the different data sources
* ucm.py: This program runs the Unobserved components model as described in the publication to extract the sea-level signal
* breakpoint.py: Script for finding a breakpoint by fitting a pice-wise linear function with the Markov-Chain Monte Carlo method with a certain start- and end-year

Additionally, the shell script `simulate.sh` runs automatically a for loop over different combinations of start- and end-years
