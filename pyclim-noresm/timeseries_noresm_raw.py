#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Friday October 28 17:08:18 2021

@author: Ada Gjermundsen

Calculates timeseries of annual, global mean variables for atmosphere and ocean. Time is given in years

Sea ice variables (area and extent) are calculated for March and September 
(Please note that time is given for March and September such that time series for March will have nans in September and vica versa)
If you use xarray for plotting, that will work just fine 
"""


import sys
# path to the piclim-noresm folder
sys.path.insert(1, '~/pyclim-NorESM/pyclim-noresm/')
from reading_routines_noresm import make_filelist_raw, read_noresm_raw
import general_util_funcs as guf
import xarray as xr
xr.set_options(enable_cftimeindex=True)
from dask.diagnostics import ProgressBar
import warnings
warnings.simplefilter('ignore')


if __name__ == '__main__':
    expid = 'NCO2x4frc2_f09_tn14_keyclim_snow'
    path = '/cluster/work/users/adagj/archive/'
    outdir = ''



    # make ocean and sea ice timeseries
    varlist = ['dp','sst','sss','mmflxd', 'mhflx', 'fice', 'temp', 'saln', 'tempga', 'salnga', 'sssga', 'sstga']
    fnames = make_filelist_raw(expid, path, component='ocean', yrs = 11, yre = 16)
    ds = read_noresm_raw(fnames, dim='time', transform_func=lambda ds: ds[varlist].map(guf.consistent_naming))
    #areaavg =ds[['sss', 'sst']].map(guf.areaavg_ocn, cmor=False).map(guf.yearly_avg)
    areaavg_masked = ds[['sss', 'sst']].map(guf.regionalavg_ocn, lat_low=-90, lat_high=-35, cmor = False).map(guf.yearly_avg)
    yravg = ds[['sssga', 'sstga', 'salnga','tempga']].map(guf.yearly_avg)
    volumeavg = ds[['temp','saln']].map(guf.volumeavg_ocn,dp = ds.dp, cmor = False).map(guf.yearly_avg)
    amoc = guf.amoc(ds['mmflxd']).map(guf.yearly_avg)
    oht = guf.atl_hfbasin(ds['mhflx']).map(guf.yearly_avg)
    siext = guf.sea_ice_ext(ds['fice'],cmor = False)
    siarea = guf.sea_ice_area(ds['fice'], cmor = False)
    combined = xr.merge([areaavg_masked, amoc, oht, siext, siarea, yravg, volumeavg])
    print(combined)
    tmp = combined.to_netcdf(outdir + expid + 'timeseries.nc', compute = False)
    with ProgressBar():
         result = tmp.compute()

