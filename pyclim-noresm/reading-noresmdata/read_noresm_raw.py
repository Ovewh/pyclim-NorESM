#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 17:08:18 2020

@author: adag
"""
import glob
import time
import numpy as np
import warnings
warnings.simplefilter('ignore')
import xarray as xr
xr.set_options(enable_cftimeindex=True)


def load_files():
    '''
    Using the h5netcdf package by passing engine='h5netcdf' to open_dataset() can sometimes be quicker than the default engine='netcdf4' that uses the netCDF4 package.

    xr.open_mfdataset('my/files/*.nc', concat_dim="time", combine="nested",
                  data_vars='minimal', coords='minimal', compat='override')
    This command concatenates variables along the "time" dimension, but only those that already contain the "time" dimension (data_vars='minimal', coords='minimal'). 
    Variables that lack the "time" dimension are taken from the first dataset (compat='override').
    '''



def read_netcdfs(files, dim='time', transform_func=None):
    def process_one_path(path):
        # use a context manager, to ensure the file gets closed after use
        with xr.open_dataset(path) as ds:
            # transform_func should do some sort of selection or
            # aggregation
            if transform_func is not None:
                ds = transform_func(ds)
            ds = ds['sst']
            # load all data from the transformed dataset, to ensure we can
            # use it after closing each original file
            ds.load()
            return ds

    paths = sorted(glob(files))
    datasets = [process_one_path(p) for p in paths]
    combined = xr.concat(datasets, dim)
    return combined

# here we suppose we only care about the combined mean of each file;
# you might also use indexing operations like .sel to subset datasets
# combined = read_netcdfs('/all/my/files/*.nc', dim='time',
#                        transform_func=None)#lambda ds: ds.mean())

def read_netcdfs_dask(files, dim='time', transform_func=None):
    with xr.open_mfdataset(sorted(glob(files)), chunks={"time": 12}, parallel=True, concat_dim="time", combine="nested",
                  data_vars='minimal', coords='minimal', compat='override') as ds:
        # transform_func should do some sort of selection or
        # aggregation
        if transform_func is not None:
            ds = transform_func(ds)
        # load all data from the transformed dataset, to ensure we can
        # use it after closing each original file
        da = ds['sst']
    return da


# here we suppose we only care about the combined mean of each file;
# you might also use indexing operations like .sel to subset datasets

filenames = '/cluster/work/users/adagj/archive/NCO2x4frc2_f09_tn14_keyclim_snow/ocn/hist/NCO2x4frc2_f09_tn14_keyclim_snow.blom.hm.00*.nc'
start = time.time()
print("Start reading netcdf files using xarray")
combined = read_netcdfs(filenames, dim='time',
                        transform_func=None)#lambda ds: ds.mean())
end = time.time()
print('Time spent in function')
print(end - start)


start = time.time()
print("Start reading netcdf files using dask and parallel")

combined = read_netcdfs_dask(filenames, dim='time',
                        transform_func=None)#lambda ds: ds.mean())
print('Time spent in function')
print(end - start)

