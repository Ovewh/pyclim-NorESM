pyclim-NorESM
Python scripts used for analysis and plotting output from the Norwegian Earth System Model (NorESM)

## Structure
One folder for each published figure. The folder contains analysis script and plotting rutines needed to reproduce the figure, except the radiative feedbacks for which we used the scripts of https://github.com/apendergrass/cam5-kernels. 

- Read NorESM data. Choose between noresm raw data or cmorized files
 
- Functions used for calculating time, regional and global averages, extract sub-regions, regridding etc.

 

- For the regridding package xesmf to work properly, you need to make a new conda environment before installing the packages:

```
  (base)$ conda create -n xesmf_env
  (base)$ conda activate xesmf_env
  (xesmf_env)$ conda install -c anaconda xarray
  (xesmf_env)$ conda install -c conda-forge esmpy scipy dask netCDF4
  (xesmf_env)$ conda install -c conda-forge xesmf
```


