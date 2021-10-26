# pyclim-NorESM

Python scripts used for analysis and plotting output from the Norwegian Earth System Model (NorESM)

## Structure
 
- Reading NorESM data. Choose between noresm raw data or cmorized files
 
- Functions used for calculating time, regional and global averages, extract sub-regions, regridding etc.

- Plotting routines: 
 

- For the regridding package xesmf to work properly, you need to make a new conda environment before installing the packages:

```
  (base)$ conda create -n xesmf_env
  (base)$ conda activate xesmf_env
  (xesmf_env)$ conda install -c anaconda xarray
  (xesmf_env)$ conda install -c conda-forge esmpy scipy dask netCDF4
  (xesmf_env)$ conda install -c conda-forge xesmf
```


